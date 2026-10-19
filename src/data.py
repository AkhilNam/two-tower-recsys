"""MovieLens-1M download, preprocessing and temporal split."""

import os
import zipfile
import urllib.request

import numpy as np
import pandas as pd

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


def download(data_dir: str = "data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, "ml-1m.zip")
    if not os.path.exists(os.path.join(data_dir, "ml-1m", "ratings.dat")):
        if not os.path.exists(zip_path):
            print(f"Downloading {ML1M_URL} ...")
            urllib.request.urlretrieve(ML1M_URL, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(data_dir)
    return os.path.join(data_dir, "ml-1m")


class RecData:
    """Container for the preprocessed dataset."""

    def __init__(self, train, val, test, genre_matrix, titles, item_mean_rating,
                 user2raw, item2raw, item_pop):
        self.train = train            # DataFrame: user, item, ts
        self.val = val
        self.test = test
        self.genre_matrix = genre_matrix  # (n_items, n_genres) float32 multi-hot
        self.titles = titles              # list[str], index-aligned
        self.item_mean_rating = item_mean_rating  # np.array (n_items,)
        self.user2raw = user2raw
        self.item2raw = item2raw
        self.item_pop = item_pop          # np.array (n_items,) train interaction counts

    @property
    def n_users(self):
        return int(max(self.train.user.max(), self.val.user.max(), self.test.user.max())) + 1

    @property
    def n_items(self):
        return self.genre_matrix.shape[0]

    def histories(self, split="train"):
        """user -> np.array of item indices, most recent last."""
        df = {"train": self.train, "val": self.val, "test": self.test}[split]
        return {u: g.sort_values("ts").item.to_numpy() for u, g in df.groupby("user")}


def load(data_dir: str = "data", min_rating: float = 4.0, min_pos: int = 3) -> RecData:
    root = download(data_dir)
    ratings = pd.read_csv(
        os.path.join(root, "ratings.dat"), sep="::", engine="python",
        names=["user", "item", "rating", "ts"],
    )
    movies = pd.read_csv(
        os.path.join(root, "movies.dat"), sep="::", engine="python",
        names=["item", "title", "genres"], encoding="latin-1",
    )

    # Implicit feedback: a rating >= min_rating counts as a positive interaction.
    pos = ratings[ratings.rating >= min_rating].copy()

    # Keep users with enough positives for a train/val/test split.
    counts = pos.groupby("user").size()
    keep_users = counts[counts >= min_pos].index
    pos = pos[pos.user.isin(keep_users)].copy()

    # Contiguous remapping.
    user_ids = np.sort(pos.user.unique())
    item_ids = np.sort(pos.item.unique())
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {m: i for i, m in enumerate(item_ids)}
    pos["user"] = pos.user.map(user_map)
    pos["item"] = pos.item.map(item_map)
    item2raw = {i: m for m, i in item_map.items()}
    user2raw = {i: u for u, i in user_map.items()}

    # Per-user temporal split: newest positive -> test, second newest -> val.
    pos = pos.sort_values(["user", "ts", "item"])
    pos["rank_desc"] = pos.groupby("user").cumcount(ascending=False)
    test = pos[pos.rank_desc == 0].drop(columns="rank_desc")
    val = pos[pos.rank_desc == 1].drop(columns="rank_desc")
    train = pos[pos.rank_desc >= 2].drop(columns="rank_desc")

    # Genre multi-hot, index-aligned.
    genre_vocab = sorted({g for gs in movies.genres for g in gs.split("|")})
    g_idx = {g: i for i, g in enumerate(genre_vocab)}
    genre_matrix = np.zeros((len(item_ids), len(genre_vocab)), dtype=np.float32)
    titles = [""] * len(item_ids)
    m_by_raw = movies.set_index("item")
    for raw, i in item_map.items():
        row = m_by_raw.loc[raw]
        titles[i] = row.title
        for g in row.genres.split("|"):
            genre_matrix[i, g_idx[g]] = 1.0

    # Item mean rating over all ratings (regression-style side info).
    item_mean = ratings.groupby("item").rating.mean()
    item_mean_rating = np.array(
        [item_mean.get(raw, ratings.rating.mean()) for raw in item_ids], dtype=np.float32)

    item_pop = np.zeros(len(item_ids), dtype=np.float32)
    pop_counts = train.groupby("item").size()
    for it, c in pop_counts.items():
        item_pop[it] = c

    stats = dict(n_users=len(user_ids), n_items=len(item_ids),
                 train_interactions=len(train), val_interactions=len(val),
                 test_interactions=len(test))
    print("Dataset:", stats)
    return RecData(train, val, test, genre_matrix, titles, item_mean_rating,
                   user2raw, item2raw, item_pop)
