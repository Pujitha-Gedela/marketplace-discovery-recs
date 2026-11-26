# Product Decision Memo: Marketplace Discovery & Recommendation Strategy

**To:** Marketplace Product & Engineering Leadership  
**From:** Product Analytics  
**Date:** November 26, 2026  
**Subject:** Recommendation Rollout Strategy & Online A/B Experiment Design  

---

## 1. Executive Summary & Shipping Recommendation
**Recommendation:** Ship a **two-tier, segment-routed hybrid recommender** combining **Popularity** (zero-history arrivals) and **Item-to-Item Collaborative Filtering (Cosine Similarity)** (in-session browsers and returning users), rather than deploying user-ID latent factor personalization (SVD/ALS) or deep learning.

* **Core Marketplace Bottleneck:** Across **1,407,580** unique visitors in the Retail Rocket dataset (2.76M events), **99.76%** view products, **2.69%** add an item to their cart, and **31.07%** of cart adders complete a purchase (**0.83%** overall view-to-purchase conversion). Lifting top-of-funnel product discovery (`view -> addtocart`) represents our highest-leverage conversion opportunity.
* **Why Hybrid Beats Pure Personalization:** **71.15%** of marketplace visitors record only a single event, and **89.26%** of test-window converters had zero prior historical interactions. Offline user-embedding models (SVD) fail on cold-start traffic (`Precision@10 = 0.00064`) and underperform Item-to-Item CF even among heavy browsers (`NDCG@10 = 0.0127` vs. `0.0255`).
* **Expected Offline-to-Online Impact Range:** Based on offline ranking gains—particularly the **20.2x NDCG@10 lift** on Product Detail Pages (PDP) for new browsers and **3.5x Precision@10 lift** for returning users—we estimate a **+8% to +18% relative lift in recommendation-attributed add-to-cart rate** and a **+3% to +7% relative lift in overall session conversion** (assuming ~30–40% of active browsers engage with recommendation carousels).

> **Critical Epistemic Caveat:** Offline evaluation measures how accurately a model predicts historical future interactions (`addtocart` or `transaction` within 7 days) under static exposure. **Offline metrics do not prove incremental revenue or conversion lift**, as historical logs suffer from presentation bias and cannot capture cannibalization of organic search/navigation. An online randomized controlled trial (RCT) is required before full rollout.

---

## 2. Offline Benchmark Evidence & Segment Breakdown (Top-10, 7-Day Intent Hit Window)
Models were evaluated on a strict **80/20 temporal split** (30-day test window; hashed item properties respected by restricting metadata to `categoryid` for diversity tracking).

| User Segment | Evaluated Converters | Winning Approach | Precision@10 | Recall@10 | NDCG@10 | MRR@10 | Catalog Coverage | Category Diversity |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **New Users (0 train events, 1st-view anchor)** | 2,494 | **Item-to-Item CF** | **0.01431** *(vs 0.00064 Pop)* | **0.11780** *(vs 0.00372 Pop)* | **0.07900** *(vs 0.00391 Pop)* | **0.07177** *(vs 0.00553 Pop)* | **5.14%** *(vs 0.005%)* | 0.69 |
| **Returning: Cold-Start (1 train event)** | 98 | **Popularity Baseline** | 0.00204 *(tie)* | 0.01531 *(tie)* | **0.01646** *(vs 0.00989 CF)* | **0.02041** *(vs 0.01191 CF)* | 0.005% | **0.90** |
| **Returning: Light Browsers (2–4 events)** | 81 | **Hybrid / Popularity** | **0.00741** *(CF)* | **0.04321** *(Pop)* | **0.04461** *(Pop)* | **0.04938** *(Pop)* | 0.33% | 0.66–0.90 |
| **Returning: Heavy Browsers (>=5 events)** | 148 | **Item-to-Item CF** | **0.00878** *(vs 0.00000 Pop)* | **0.02618** *(vs 0.00000 Pop)* | **0.02546** *(vs 0.00000 Pop)* | **0.03153** *(vs 0.00000 Pop)* | **0.51%** | 0.71 |
| **Returning: All Combined** | 327 | **Item-to-Item CF** | **0.00642** | **0.02408** | **0.01906** | **0.02310** | **1.06%** | 0.70 |

### Where Each Model Succeeds & Fails (Honest Assessment)
1. **Popularity is genuinely hard to beat in low-signal states:** For returning users with only 1–4 historical interactions, global weighted popularity beats both Item-to-Item CF and SVD on top-slot ranking (`NDCG@10 = 0.0446` vs. `0.0185` for light browsers) and offers maximum category breadth (`0.90` diversity).
2. **Popularity completely fails heavy browsers (`0.0000` across all metrics):** Users with $\ge 5$ interactions have already bypassed mainstream bestsellers; showing static popular items yields zero hits.
3. **In-session Item-to-Item CF unlocks cold-start discovery:** While offline user-ID models cannot score new visitors, conditioning Item-to-Item CF on a visitor's **first viewed item** delivers our strongest overall performance (`NDCG@10 = 0.0790`, `MRR@10 = 0.0718`) and expands catalog exposure **>1,000x** over popularity.
4. **Why we reject Personalized SVD:** Sparse interaction matrices (mean `1.96` events/user) cause latent factor models to underperform Item-to-Item CF across every segment while adding matrix-retraining latency and cold-start complexity.

---

## 3. Production Routing Architecture ("Who Gets What")
* **Surface 1 — Homepage / Zero-History Landing (`0 session views, <5 historical events`):** Serve **Intent-Weighted Popularity** (ranked by `5*purchase + 3*cart + 1*view` over a rolling 14-day window).
* **Surface 2 — Product Detail Page & Post-Click Modules (`>=1 in-session view`):** Serve **Real-Time Item-to-Item CF** seeded on the current session's viewed items, excluding already-purchased SKUs.
* **Surface 3 — Returning Power Users (`>=5 historical events`):** Serve **Multi-Item History Item-to-Item CF** blended with recent session context.

---

## 4. Online A/B Test Proposal

### Experimental Setup
* **Randomization Unit:** `visitorid` (persistent cookie/account ID) split **50% Control** (existing Popularity recommendations) vs. **50% Treatment** (Segment-Routed Hybrid Popularity + Item-to-Item CF), stratified by new vs. returning visitor status.
* **Primary Metric:** **Recommendation-Driven Conversion Rate** — percentage of exposed visitors who click a recommended item and complete an `addtocart` or `transaction` on that item within 24 hours.
* **Secondary Metrics:**
  * Overall sitewide `view -> addtocart` rate (baseline `2.69%`) and `view -> transaction` rate (baseline `0.83%`).
  * Recommendation Carousel Click-Through Rate (CTR) and Average Order Value (AOV).
  * Long-tail catalog discovery share (% of orders coming from items outside the top 1% bestsellers).
* **Guardrail Metrics:**
  * **Page Load / API P99 Latency:** Must not degrade by `> +25 ms` (precomputing top-50 cosine neighbors per item in Redis keeps lookup `< 5 ms`).
  * **Cart Abandonment Rate:** Ensure relevance clustering does not induce low-intent accidental cart adds (must not increase by `> +1.0%` relative).
  * **Out-of-Stock Click Rate:** Filter items where `available == 0` in `item_properties` prior to serving.

### Sample Size & Duration Calculation
* **Baseline Conversion:** Using sitewide visitor purchase conversion $p_0 = 0.83\%$ (`0.0083`) as a conservative lower bound (or `2.69%` for add-to-cart).
* **Minimum Detectable Effect (MDE):** **+10% relative lift** on overall conversion ($p_1 = 0.913\%$, absolute lift $\Delta = 0.083\%$), with two-sided $\alpha = 0.05$ ($Z_{1-\alpha/2} = 1.96$) and statistical power $1 - \beta = 0.80$ ($Z_{0.80} = 0.84$).
* **Required Sample Size:**
  $$n = \frac{2 \cdot \bar{p}(1 - \bar{p}) \cdot (Z_{1-\alpha/2} + Z_{0.80})^2}{\Delta^2} \approx \frac{2 \cdot 0.0087 \cdot 0.9913 \cdot (2.80)^2}{(0.00083)^2} \approx \mathbf{196,000\text{ visitors per variant}}$$
  *(Total required sample = **~392,000 visitors**; if evaluating directly on the `2.69%` add-to-cart primary funnel, required sample drops to **~59,500 visitors per variant**.)*
* **Experiment Duration:** At Retail Rocket's historical traffic rate (~10,350 unique visitors/day across 136 days), reaching 392,000 visitors requires **~38 days**. Running for **6 full weeks (42 days)** captures complete weekly shopping cycles and accounts for returning-user attribution lag.

### Key Risks & Mitigations
1. **Filter Bubble / Narrow Diversity:** Item-to-Item CF reduces category diversity from `0.90` to `0.69`. *Mitigation:* Reserve slots #9–#10 in the top-10 carousel for high-converting complementary/popular items when diversity falls below `0.50`.
2. **Stale Similarity Matrix on New Arrivals:** Newly listed SKUs lack co-visitation history. *Mitigation:* Fall back to category-level popularity (`categoryid`) until an item accumulates $\ge 10$ views, and refresh cosine similarity matrices nightly.
3. **Novelty Effect:** Returning heavy browsers may click personalized recommendations initially out of novelty. *Mitigation:* Enforce a minimum 4-week runtime and monitor week-over-week cohort retention before declaring a winner.