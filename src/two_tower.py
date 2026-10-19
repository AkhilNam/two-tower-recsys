"""Two-tower retrieval model with in-batch sampled softmax.

User tower: mean-pooled ID embeddings of the user's training history, through an MLP.
Item tower: item ID embedding + multi-hot genre projection, through an MLP.
Loss: in-batch sampled softmax with a count-based log-uniform correction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, n_items: int, n_genres: int, dim: int = 64, hidden: int = 128):
        super().__init__()
        self.dim = dim
        self.item_emb = nn.Embedding(n_items, dim)
        nn.init.normal_(self.item_emb.weight, std=0.05)
        self.genre_proj = nn.Linear(n_genres, dim, bias=False)
        self.item_mlp = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.user_mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(10.0), dtype=torch.float32))

    def item_vector(self, item_idx, genre_multi):
        h = torch.cat([self.item_emb(item_idx), self.genre_proj(genre_multi)], dim=-1)
        return F.normalize(self.item_mlp(h), dim=-1)

    def user_vector_from_hist(self, hist_emb_mean):
        return F.normalize(self.user_mlp(hist_emb_mean), dim=-1)


def history_mean_emb(model, hist_items_padded, hist_mask):
    """hist_items_padded: (B, L) long, hist_mask: (B, L) float -> (B, dim)"""
    emb = model.item_emb(hist_items_padded)            # (B, L, dim)
    emb = emb * hist_mask.unsqueeze(-1)
    denom = hist_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    return emb.sum(dim=1) / denom


def train_two_tower(data, dim=64, hidden=128, epochs=15, batch_size=1024,
                    lr=1e-3, max_hist=50, seed=42, device="cpu"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    genre_t = torch.tensor(data.genre_matrix)
    model = TwoTower(data.n_items, data.genre_matrix.shape[1], dim, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    histories = data.histories("train")
    users = np.array(sorted(histories.keys()))
    # (user, positive) pairs
    train = data.train
    pairs = list(zip(train.user.to_numpy(), train.item.to_numpy()))

    counts = data.item_pop.copy()
    counts[counts == 0] = 1.0
    log_q = torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32)

    def make_batch(batch_pairs):
        users_b = np.array([p[0] for p in batch_pairs])
        items_b = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long)
        hists = [histories[u][-max_hist:] for u in users_b]
        L = max(len(h) for h in hists)
        pad = np.zeros((len(hists), L), dtype=np.int64)
        mask = np.zeros((len(hists), L), dtype=np.float32)
        for i, h in enumerate(hists):
            pad[i, : len(h)] = h
            mask[i, : len(h)] = 1.0
        return items_b, torch.tensor(pad), torch.tensor(mask)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        total_loss, nb = 0.0, 0
        for s in range(0, len(pairs), batch_size):
            batch_pairs = pairs[s: s + batch_size]
            items_b, pad, mask = make_batch(batch_pairs)
            u_mean = history_mean_emb(model, pad, mask)
            u_vec = model.user_vector_from_hist(u_mean)
            i_vec = model.item_vector(items_b, genre_t[items_b])
            logits = (u_vec @ i_vec.T) * model.logit_scale.exp()
            logits = logits - log_q[items_b]           # sampled-softmax correction
            labels = torch.arange(len(batch_pairs))
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            nb += 1
        print(f"epoch {epoch + 1}/{epochs}  loss {total_loss / nb:.4f}")
    return model


@torch.no_grad()
def all_item_vectors(model, data, batch=2048):
    model.eval()
    genre_t = torch.tensor(data.genre_matrix)
    vecs = []
    for s in range(0, data.n_items, batch):
        idx = torch.arange(s, min(s + batch, data.n_items))
        vecs.append(model.item_vector(idx, genre_t[idx]).numpy())
    return np.concatenate(vecs)


@torch.no_grad()
def user_vectors(model, data, users, max_hist=50):
    model.eval()
    histories = data.histories("train")
    out = np.zeros((len(users), model.dim), dtype=np.float32)
    for j, u in enumerate(users):
        h = histories[u][-max_hist:]
        emb = model.item_emb(torch.tensor(h)).mean(dim=0, keepdim=True)
        out[j] = model.user_vector_from_hist(emb).numpy()[0]
    return out
