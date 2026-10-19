"""Print top-10 recommendations for a raw MovieLens user id.

Usage: python recommend.py --user 1
"""

import argparse
import pickle

import numpy as np
import torch

from src.two_tower import TwoTower


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int, required=True, help="raw MovieLens-1M user id")
    args = ap.parse_args()

    with open("artifacts/model.pkl", "rb") as f:
        art = pickle.load(f)
    raw2user = {r: i for i, r in art["user2raw"].items()}
    if args.user not in raw2user:
        raise SystemExit(f"user {args.user} not in dataset")
    u = raw2user[args.user]

    model = TwoTower(art["n_items"], art["n_genres"])
    model.load_state_dict(art["state_dict"])
    model.eval()

    genre_t = torch.tensor(art["genre_matrix"])
    with torch.no_grad():
        item_vecs = model.item_vector(torch.arange(art["n_items"]), genre_t).numpy()
        hist = art["histories"][u][-50:]
        emb = model.item_emb(torch.tensor(hist)).mean(dim=0, keepdim=True)
        u_vec = model.user_vector_from_hist(emb).numpy()[0]
    scores = u_vec @ item_vecs.T
    scores[hist] = -np.inf  # hide what they already liked
    top = np.argsort(-scores)[:10]
    print(f"Top-10 for user {args.user} (liked {len(hist)} movies in train):")
    for rank, it in enumerate(top, 1):
        print(f"{rank:2d}. {art['titles'][it]}  (score {scores[it]:.3f})")


if __name__ == "__main__":
    main()
