"""Non-neural baselines: popularity and item-item cosine KNN."""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def train_matrix(train, n_users, n_items):
    return csr_matrix(
        (np.ones(len(train), dtype=np.float32), (train.user.to_numpy(), train.item.to_numpy())),
        shape=(n_users, n_items),
    )


class Popularity:
    def __init__(self, item_pop):
        self.scores = item_pop.astype(np.float64)

    def score_users(self, users):
        return np.tile(self.scores, (len(users), 1))


class ItemKNN:
    """Cosine item-item KNN over the implicit train matrix."""

    def __init__(self, k: int = 200):
        self.k = k

    def fit(self, train_mat):
        item_norm = normalize(train_mat.T, axis=1)          # (n_items, n_users)
        sim = (item_norm @ item_norm.T).tocsr()             # item-item cosine
        # keep top-k per row
        sim = sim.tolil()
        for i in range(sim.shape[0]):
            row = sim.rows[i]
            data = sim.data[i]
            if len(row) > self.k + 1:
                keep = np.argpartition(-np.array(data), self.k + 1)[: self.k + 1]
                sim.rows[i] = [row[j] for j in keep]
                sim.data[i] = [data[j] for j in keep]
        self.sim = sim.tocsr()
        return self

    def score_users(self, users, train_mat):
        return (train_mat[users] @ self.sim).toarray()
