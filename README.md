# Marketplace Discovery & Recommendation System

An end-to-end **Product Analytics & Applied Recommendation** case study built on the [Retail Rocket e-commerce dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) (2.76M behavioral events across 1.41M visitors and 235K products).

## Business Problem & Key Findings
* **Funnel Bottleneck:** While **99.76%** of visitors browse products, only **2.69%** add an item to their cart—yet **31.07%** of cart adders purchase (**0.83%** overall conversion).
* **Cold-Start Reality:** **71.15%** of visitors have only 1 event, and **89.26%** of test-period converters are first-time users unseen in training.
* **Model Verdict:** A **Segment-Routed Hybrid Strategy** (Popularity for zero-history arrivals + Session/History Item-to-Item Collaborative Filtering after the first view) outperforms both standalone Popularity and Personalized Matrix Factorization (SVD):
  * **New Users (after 1st view):** Item-to-Item CF delivers a **20.2x lift in NDCG@10** (`0.0790` vs. `0.0039`) and expands catalog coverage from `0.005%` to `5.14%`.
  * **Heavy Browsers ($\ge 5$ events):** Popularity scores `0.0000` across all accuracy metrics, whereas Item-to-Item CF achieves `NDCG@10 = 0.0255` and `MRR@10 = 0.0315`.
  * **Light / 1-Event Returning Browsers:** Popularity remains competitive or superior on ranking quality (`NDCG@10 = 0.0446` vs. `0.0185`), justifying a hybrid routing rule rather than a single algorithm.

## Repository Structure
```text
marketplace-discovery-recs/
├── app.py                                      # Interactive Streamlit dashboard
├── notebooks/
│   ├── 01_eda_and_funnel.ipynb                 # Schema audit, hashed property check & funnel analysis
│   └── 02_model_benchmark_and_segments.ipynb   # Temporal split, model training & segment evaluation
├── src/
│   ├── data_prep.py                            # Leak-free temporal splitting & user segmentation
│   ├── models.py                               # Popularity, Item-to-Item CF, and Personalized SVD
│   └── metrics.py                              # Precision, Recall, NDCG, MAP, MRR, Coverage, Diversity
├── reports/
│   └── product_decision_memo.md                # 1-2 page executive memo & A/B test proposal
└── data/                                       # Local dataset & processed benchmark CSV (gitignored)