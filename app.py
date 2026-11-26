import pandas as pd
import streamlit as st

st.set_page_config(page_title="Marketplace Discovery & Recs", layout="wide")

st.title("Marketplace Discovery & Recommendation Engine — Product Analytics")
st.caption("Dataset: Retail Rocket (2.76M events | 1.41M visitors | 80/20 temporal split | 7-day intent hit window)")

# Load precomputed evaluation results
df = pd.read_csv("data/processed/evaluation_results.csv")

# Top-level marketplace funnel KPIs
st.subheader("1. Marketplace Conversion Funnel & Sparsity Baseline")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique Visitors", "1,407,580", "71.15% single-event")
c2.metric("View -> Add-to-Cart", "2.69%", "Primary discovery bottleneck")
c3.metric("Add-to-Cart -> Purchase", "31.07%", "Strong checkout conversion")
c4.metric("Overall View -> Purchase", "0.83%", "11,719 buyers")

st.divider()

# Interactive Segment Explorer
st.subheader("2. Offline Model Benchmark by User Segment (Top-10 Recommendations)")
segments = df["Segment"].unique().tolist()
selected_segment = st.selectbox("Select User Cohort / Activity Segment:", segments, index=0)

sub = df[df["Segment"] == selected_segment].copy()
users_n = int(sub["Users_Evaluated"].iloc[0])
st.markdown(f"**Evaluated Converters in Cohort:** `{users_n:,}` users (target = `addtocart` or `transaction` within 7 days)")

# Format display table
display_cols = [
    "Model", "Precision@10", "Recall@10", "NDCG@10",
    "MAP@10", "MRR@10", "Catalog_Coverage", "Category_Diversity"
]
st.dataframe(
    sub[display_cols].style.format({
        "Precision@10": "{:.5f}",
        "Recall@10": "{:.5f}",
        "NDCG@10": "{:.5f}",
        "MAP@10": "{:.5f}",
        "MRR@10": "{:.5f}",
        "Catalog_Coverage": "{:.4%}",
        "Category_Diversity": "{:.2f}",
    }).highlight_max(subset=["Precision@10", "Recall@10", "NDCG@10", "MAP@10", "MRR@10", "Catalog_Coverage"], color="#1e4620"),
    width="stretch",
    hide_index=True,
)

# Visual Comparison Charts
col_left, col_right = st.columns(2)
with col_left:
    st.markdown("**Ranking Quality (NDCG@10 & MRR@10)**")
    chart_rank = sub.set_index("Model")[["NDCG@10", "MRR@10"]]
    st.bar_chart(chart_rank)

with col_right:
    st.markdown("**Catalog Reach vs. Category Diversity**")
    chart_div = sub.set_index("Model")[["Catalog_Coverage", "Category_Diversity"]]
    st.bar_chart(chart_div)

st.divider()
st.subheader("3. Recommended Hybrid Routing Architecture")
st.info(
    "**Ship a Segment-Routed Hybrid Strategy (No Deep Learning / SVD Overhead):**\n"
    "- **0 Prior Events (Landing Page):** Global / Category Popularity (`NDCG@10 = 0.0165` on 1-event users).\n"
    "- **>=1 In-Session View (Product Detail Page):** Real-time Item-to-Item CF (`NDCG@10 = 0.0790`, **20x lift** over popularity).\n"
    "- **Returning Heavy Browsers (>=5 Events):** Multi-item History Item-to-Item CF (`MRR@10 = 0.0315` vs. `0.0000` for popularity)."
)