import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

EVENT_WEIGHTS = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}


class RecommenderSuite:
    def __init__(self, n_factors: int = 64, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state

    def fit(self, train_df: pd.DataFrame):
        df = train_df.copy()
        df["weight"] = df["event"].map(EVENT_WEIGHTS).fillna(1.0)

        # 1. Popularity baseline (weighted by intent: transaction > cart > view)
        pop_scores = df.groupby("itemid")["weight"].sum().sort_values(ascending=False)
        self.popular_items = pop_scores.index.tolist()
        self.all_train_items = set(self.popular_items)

        # Build sparse User x Item interaction matrix
        agg = df.groupby(["visitorid", "itemid"], as_index=False)["weight"].sum()
        self.user_ids = agg["visitorid"].unique()
        self.item_ids = agg["itemid"].unique()
        self.user_to_idx = {u: i for i, u in enumerate(self.user_ids)}
        self.item_to_idx = {it: i for i, it in enumerate(self.item_ids)}

        rows = agg["visitorid"].map(self.user_to_idx).values
        cols = agg["itemid"].map(self.item_to_idx).values
        vals = np.log1p(agg["weight"].values)  # confidence scaling for implicit feedback

        self.ui_matrix = csr_matrix(
            (vals, (rows, cols)),
            shape=(len(self.user_ids), len(self.item_ids)),
            dtype=np.float32,
        )

        # 2. Item-to-Item Collaborative Filtering (L2-normalized item vectors for fast cosine dot-products)
        self.item_vecs_norm = normalize(self.ui_matrix.T.tocsr(), norm="l2", axis=1)

        # 3. Personalized Latent Factor Model (SVD on implicit confidence matrix)
        k = min(self.n_factors, min(self.ui_matrix.shape) - 1)
        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.user_factors = svd.fit_transform(self.ui_matrix)
        self.item_factors = svd.components_.T  # shape: (n_items, k)
        return self

    def recommend_popularity(self, k: int = 10) -> list[int]:
        return self.popular_items[:k]

    def recommend_item_cf(self, history_items: list[int], k: int = 10, exclude_seen: bool = True) -> list[int]:
        valid_idxs = [self.item_to_idx[it] for it in history_items if it in self.item_to_idx]
        if not valid_idxs:
            return self.recommend_popularity(k)

        # Sum cosine similarities across the user's interacted items
        seed_profile = self.item_vecs_norm[valid_idxs].sum(axis=0)
        scores = np.asarray(self.item_vecs_norm.dot(seed_profile.T)).ravel()

        if exclude_seen:
            scores[valid_idxs] = -np.inf

        top_idxs = np.argpartition(-scores, k)[:k]
        top_idxs = top_idxs[np.argsort(-scores[top_idxs])]
        recs = [int(self.item_ids[i]) for i in top_idxs if np.isfinite(scores[i])]
        if len(recs) < k:
            seen_set = set(recs) | (set(history_items) if exclude_seen else set())
            recs.extend([it for it in self.popular_items if it not in seen_set][: k - len(recs)])
        return recs[:k]

    def recommend_personalized(self, visitor_id: int, k: int = 10, exclude_seen: bool = True) -> list[int]:
        if visitor_id not in self.user_to_idx:
            return self.recommend_popularity(k)

        u_idx = self.user_to_idx[visitor_id]
        scores = self.user_factors[u_idx] @ self.item_factors.T

        if exclude_seen:
            seen_cols = self.ui_matrix[u_idx].indices
            scores = scores.copy()
            scores[seen_cols] = -np.inf

        top_idxs = np.argpartition(-scores, k)[:k]
        top_idxs = top_idxs[np.argsort(-scores[top_idxs])]
        return [int(self.item_ids[i]) for i in top_idxs][:k]