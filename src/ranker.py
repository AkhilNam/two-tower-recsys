"""Second-stage ranker: logistic regression over retrieval + side features.

Trained on validation candidates (label: candidate is the user's held-out item),
applied to re-rank the retriever's top-N test candidates.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def candidate_features(users, cand_lists, cand_scores, data, retriever_scores):
    """Build per-candidate feature rows.

    Features: retrieval score, log item popularity, item mean rating,
    genre overlap with the user's training profile, log user activity.
    """
    histories = data.histories("train")
    genre = data.genre_matrix
    feats, labels = [], []
    for j, u in enumerate(users):
        hist = histories[u]
        user_genre = genre[hist].mean(axis=0) if len(hist) else np.zeros(genre.shape[1])
        user_genre = user_genre / (np.linalg.norm(user_genre) + 1e-9)
        for it, rscore in zip(cand_lists[j], cand_scores[j]):
            g = genre[it]
            overlap = float(user_genre @ g / (np.linalg.norm(g) + 1e-9))
            feats.append([
                rscore,
                np.log1p(data.item_pop[it]),
                data.item_mean_rating[it],
                overlap,
                np.log1p(len(hist)),
            ])
    return np.array(feats, dtype=np.float32)


class Ranker:
    def __init__(self):
        self.scaler = StandardScaler()
        self.clf = LogisticRegression(max_iter=2000, C=1.0)

    def fit(self, X, y):
        self.clf.fit(self.scaler.fit_transform(X), y)
        return self

    def score(self, X):
        return self.clf.predict_proba(self.scaler.transform(X))[:, 1]
