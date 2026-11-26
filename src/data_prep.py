import pandas as pd
import numpy as np
from pathlib import Path


def load_events(raw_dir: str = "data/raw") -> pd.DataFrame:
    df = pd.read_csv(Path(raw_dir) / "events.csv")
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


def load_item_categories(raw_dir: str = "data/raw") -> dict[int, str]:
    """Extract the latest known categoryid per itemid (hashed token) for diversity metrics."""
    p1 = pd.read_csv(Path(raw_dir) / "item_properties_part1.csv")
    p2 = pd.read_csv(Path(raw_dir) / "item_properties_part2.csv")
    props = pd.concat([p1, p2], ignore_index=True)

    cats = props[props["property"] == "categoryid"].copy()
    cats = cats.sort_values("timestamp").drop_duplicates("itemid", keep="last")
    return dict(zip(cats["itemid"], cats["value"].astype(str)))


def prepare_temporal_split(
    events: pd.DataFrame,
    train_quantile: float = 0.80,
    hit_window_days: int = 7,
) -> dict:
    """
    Create leak-free temporal train/test split.
    A 'hit' is an addtocart or transaction event occurring within `hit_window_days`
    after the cutoff (for returning users) or after a new user's first test view.
    """
    cutoff_dt = events["dt"].quantile(train_quantile)
    train_df = events[events["dt"] < cutoff_dt].copy()
    test_df = events[events["dt"] >= cutoff_dt].copy()

    # Train interaction counts & user segments
    train_counts = train_df.groupby("visitorid").size()
    train_users = set(train_counts.index)

    # Filter test events to positive intent hits (addtocart / transaction)
    test_positives = test_df[test_df["event"].isin(["addtocart", "transaction"])].copy()

    # For returning users: hits within N days of cutoff_dt
    ret_mask = test_positives["visitorid"].isin(train_users)
    ret_hits_df = test_positives[
        ret_mask & (test_positives["dt"] <= cutoff_dt + pd.Timedelta(days=hit_window_days))
    ]

    # For new users: use their 1st test view as anchor, hits within N days after 1st view
    new_test = test_df[~test_df["visitorid"].isin(train_users)]
    first_view = (
        new_test[new_test["event"] == "view"]
        .sort_values("timestamp")
        .drop_duplicates("visitorid", keep="first")[["visitorid", "itemid", "dt"]]
        .rename(columns={"itemid": "seed_item", "dt": "first_view_dt"})
    )

    new_hits_merged = test_positives[~ret_mask].merge(first_view, on="visitorid", how="inner")
    new_hits_df = new_hits_merged[
        (new_hits_merged["dt"] > new_hits_merged["first_view_dt"])
        & (new_hits_merged["dt"] <= new_hits_merged["first_view_dt"] + pd.Timedelta(days=hit_window_days))
        & (new_hits_merged["itemid"] != new_hits_merged["seed_item"])
    ]

    # Combine ground truth sets: visitorid -> set of target itemids
    gt_returning = ret_hits_df.groupby("visitorid")["itemid"].apply(set).to_dict()
    gt_new = new_hits_df.groupby("visitorid")["itemid"].apply(set).to_dict()
    seed_new = dict(zip(first_view["visitorid"], first_view["seed_item"]))

    def get_activity_segment(uid: int) -> str:
        c = train_counts.get(uid, 0)
        if c == 0:
            return "New (0 train events)"
        elif c == 1:
            return "Cold-Start (1 event)"
        elif c <= 4:
            return "Light Browser (2-4 events)"
        return "Heavy Browser (>=5 events)"

    return {
        "cutoff_dt": cutoff_dt,
        "train_df": train_df,
        "test_df": test_df,
        "gt_returning": gt_returning,
        "gt_new": gt_new,
        "seed_new": seed_new,
        "get_segment": get_activity_segment,
    }