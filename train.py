"""End-to-end: data -> baselines -> two-tower -> second-stage ranker -> results."""

import json
import os
import pickle

import numpy as np
import torch

from src.data import load
from src.baselines import Popularity, ItemKNN, train_matrix
from src.two_tower import train_two_tower, all_item_vectors, user_vectors
from src.evaluate import evaluate_model, recall_ndcg
from src.ranker import candidate_features, Ranker

TOPN = 200  # retrieval candidates passed to the ranker


def main():
    os.makedirs("artifacts", exist_ok=True)
    data = load()
    users = np.sort(data.test.user.unique())
    train_mat = train_matrix(data.train, data.n_users, data.n_items)
    results = []

    # --- baselines ---
    pop = Popularity(data.item_pop)
    pop_scores = pop.score_users(users)

    knn = ItemKNN(k=200).fit(train_mat)
    knn_scores = knn.score_users(users, train_mat)

    # --- two-tower ---
    model = train_two_tower(data, epochs=10)
    item_vecs = all_item_vectors(model, data)
    tt_scores = user_vectors(model, data, users) @ item_vecs.T

    # --- retrieval eval (val + test) ---
    for name, scores in [("popularity", pop_scores),
                         ("item-knn (cosine)", knn_scores),
                         ("two-tower", tt_scores)]:
        for split in ("val", "test"):
            row, _ = evaluate_model(name, scores, data, users, split)
            results.append(row)
            print(row)

    # --- second-stage ranker, trained on val, applied to test ---
    def topn_candidates(scores):
        s = scores.copy()
        hist_map = data.histories("train")
        for j, u in enumerate(users):
            s[j, hist_map[u]] = -np.inf
        part = np.argpartition(-s, TOPN - 1, axis=1)[:, :TOPN]
        order = np.argsort(-np.take_along_axis(s, part, axis=1), axis=1)
        cands = np.take_along_axis(part, order, axis=1)
        cscores = np.take_along_axis(s, cands, axis=1)
        return cands, cscores

    val_cands, val_cs = topn_candidates(tt_scores)
    val_target = data.val.set_index("user").loc[users].item.to_numpy()
    Xv = candidate_features(users, val_cands, val_cs, data, tt_scores)
    yv = (val_cands == val_target[:, None]).astype(int).ravel()
    ranker = Ranker().fit(Xv, yv)

    test_cands, test_cs = topn_candidates(tt_scores)
    Xt = candidate_features(users, test_cands, test_cs, data, tt_scores)
    rscores = ranker.score(Xt).reshape(test_cands.shape)
    rerank = np.full_like(tt_scores, -np.inf, dtype=np.float64)
    np.put_along_axis(rerank, test_cands, rscores, axis=1)

    row, _ = evaluate_model("two-tower + ranker", rerank, data, users, "test")
    results.append(row)
    print(row)

    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("artifacts/model.pkl", "wb") as f:
        pickle.dump({
            "state_dict": model.state_dict(),
            "n_items": data.n_items,
            "n_genres": data.genre_matrix.shape[1],
            "genre_matrix": data.genre_matrix,
            "titles": data.titles,
            "user2raw": data.user2raw,
            "item2raw": data.item2raw,
            "histories": data.histories("train"),
        }, f)
    print("Saved results.json and artifacts/model.pkl")


if __name__ == "__main__":
    main()
