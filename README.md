# Two-Tower RecSys

A working two-stage movie recommender on MovieLens-1M: a neural **two-tower
retrieval** model, a logistic **second-stage ranker**, and two non-neural
baselines, all compared with the same temporal-split offline evaluation.

## What it does

- **Data**: MovieLens-1M, treated as implicit feedback (a rating >= 4 is a
  positive interaction). Per-user temporal split - each user's newest positive
  is the test item, second newest is validation, the rest is training.
- **Baselines**: item popularity, and item-item cosine KNN over the implicit
  matrix.
- **Two-tower retrieval**: the item tower combines an item ID embedding with a
  multi-hot genre projection; the user tower mean-pools ID embeddings over the
  user's recent history. Both towers end in an MLP with L2-normalized output.
  Trained with in-batch sampled softmax plus a count-based log-uniform
  correction.
- **Second-stage ranker**: logistic regression over the retriever's top-200
  candidates, using the retrieval score, log item popularity, item mean rating,
  genre overlap with the user's profile, and user activity. Trained on
  validation candidates, applied to test.
- **Metrics**: Recall@10/50, NDCG@10/50, and catalog coverage@10 over the full
  item catalog, with training items excluded.

## Results

Dataset: 6,035 users x 3,533 items; 563,206 training positives, one
validation and one test positive per user (temporal split).

Test split (held-out newest positive per user, full-catalog scoring, training
items excluded):

| Model | Recall@10 | NDCG@10 | Recall@50 | NDCG@50 | Coverage@10 |
|---|---|---|---|---|---|
| Popularity | 0.0389 | 0.0191 | 0.1278 | 0.0380 | 3.3% |
| Item-KNN (cosine) | 0.0510 | 0.0254 | 0.1728 | 0.0511 | 9.7% |
| Two-tower | 0.0650 | 0.0303 | 0.2222 | 0.0640 | 54.6% |
| Two-tower + ranker | 0.0630 | 0.0301 | 0.2270 | 0.0649 | 30.0% |

Validation split shows the same ordering (two-tower 0.0805 Recall@10 vs
item-KNN 0.0648 vs popularity 0.0456).

What the numbers say:

- The two-tower beats both baselines on every retrieval metric, and its
  top-10 lists cover 54.6% of the catalog vs 9.7% for item-KNN and 3.3% for
  popularity - it is not just re-ranking hits.
- The logistic ranker is roughly neutral here: it nudges Recall@50 up
  (0.2222 -> 0.2270) but trims Recall@10 slightly (0.0650 -> 0.0630). With
  one positive per user and five hand-built features there is little for it
  to learn; kept in the pipeline as an honest negative-ish result, not
  hidden.


All numbers come from `python train.py` on this machine; nothing is copied
from papers or other repos. Single relevant item per user, so IDCG = 1.

## Reproduce

```bash
pip install -r requirements.txt
python train.py          # downloads MovieLens-1M, trains everything, writes results.json
python recommend.py --user 1   # top-10 picks for a MovieLens user id
```

Training runs on CPU in a few minutes. `train.py` saves the model and mappings
to `artifacts/model.pkl`; `results.json` holds the metric table printed above.

## Layout

```
src/data.py       MovieLens download, implicit-feedback preprocessing, temporal split
src/baselines.py  popularity + item-item cosine KNN
src/two_tower.py  two-tower model and sampled-softmax training
src/ranker.py     second-stage logistic ranker
src/evaluate.py   Recall@K / NDCG@K / coverage
train.py          end-to-end pipeline
recommend.py      top-10 recommendations for one user
```

## Honest limitations

- MovieLens-1M is small and old; this is a methods demo, not production rec-sys.
- The user tower only sees item IDs (no text, image, or demographic features).
- Offline metrics on one held-out item per user are noisy; no online A/B here.
