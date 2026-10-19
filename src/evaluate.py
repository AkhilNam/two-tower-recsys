"""Offline retrieval metrics: Recall@K, NDCG@K, catalog coverage."""

import numpy as np


def recall_ndcg(scores, train_hist, target_item, ks=(10, 50)):
    """scores: (B, n_items). Exclude train items. One held-out target per user."""
    B = scores.shape[0]
    for j, u_items in enumerate(train_hist):
        scores[j, u_items] = -np.inf
    max_k = max(ks)
    top = np.argpartition(-scores, max_k - 1, axis=1)[:, :max_k]
    # full sort within top-k for correct NDCG ordering
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    ranked = np.take_along_axis(top, order, axis=1)

    out = {}
    targets = np.array(target_item)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    for k in ks:
        hits = (ranked[:, :k] == targets[:, None])
        out[f"recall@{k}"] = float(hits.any(axis=1).mean())
        dcg = (hits * discounts[:k]).sum(axis=1)
        out[f"ndcg@{k}"] = float(dcg.mean())  # IDCG = 1 (single relevant item)
    out["coverage@10"] = float(len(np.unique(ranked[:, :10])) / scores.shape[1])
    return out, ranked


def evaluate_model(name, scores, data, users, split="test", ks=(10, 50)):
    held = data.val if split == "val" else data.test
    target = held.set_index("user").loc[users].item.to_numpy()
    train_hist = [data.histories("train")[u] for u in users]
    metrics, ranked = recall_ndcg(scores.copy(), train_hist, target, ks)
    row = {"model": name, "split": split, **{k: round(v, 4) for k, v in metrics.items()}}
    return row, ranked
