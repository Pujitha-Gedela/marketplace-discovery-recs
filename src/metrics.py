import numpy as np
import pandas as pd


def evaluate_recommendations(
    recs_by_user: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    total_catalog_items: int,
    item_to_cat: dict[int, str],
    k: int = 10,
) -> dict[str, float]:
    precisions, recalls, ndcgs, maps, mrrs, diversities = [], [], [], [], [], []
    recommended_unique = set()

    for uid, actual in ground_truth.items():
        if not actual or uid not in recs_by_user:
            continue
        preds = recs_by_user[uid][:k]
        recommended_unique.update(preds)

        hits = [1 if item in actual else 0 for item in preds]

        # Precision@K & Recall@K
        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(actual))

        # MRR@K
        rr = 0.0
        for rank, h in enumerate(hits, start=1):
            if h:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

        # MAP@K
        ap_sum, hit_count = 0.0, 0
        for rank, h in enumerate(hits, start=1):
            if h:
                hit_count += 1
                ap_sum += hit_count / rank
        maps.append(ap_sum / min(len(actual), k))

        # NDCG@K
        dcg = sum(h / np.log2(rank + 1) for rank, h in enumerate(hits, start=1))
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(actual), k) + 1))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        # Intra-list Category Diversity (unique categories in top-K / K)
        cats = [item_to_cat.get(it, f"unk_{it}") for it in preds]
        diversities.append(len(set(cats)) / k if preds else 0.0)

    return {
        "Users_Evaluated": len(precisions),
        f"Precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"Recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"NDCG@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"MAP@{k}": float(np.mean(maps)) if maps else 0.0,
        f"MRR@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
        "Catalog_Coverage": len(recommended_unique) / total_catalog_items if total_catalog_items else 0.0,
        "Category_Diversity": float(np.mean(diversities)) if diversities else 0.0,
    }
