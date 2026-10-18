"""
models.py
=========
Model zoo. Each model implements:

    fit(data)              : data is a TrainData (CSR matrix X + ordered train_df)
    scores(user_idx)       : dense np.array of length n_items
    scores_batch(user_ids) : (len(user_ids), n_items) score matrix (for fast eval)

Trainable models accept `neg_sampling` in {"uniform", "popularity"} and set
`uses_train_neg = True` so the multiverse can treat training-time negative
sampling as a degree of freedom. GPU models use torch with auto CUDA->CPU.
Add models by subclassing BaseModel and registering in MODEL_REGISTRY.
"""
from __future__ import annotations

from typing import Dict, Type

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
class BaseModel:
    name = "base"
    uses_train_neg = False        # does this model train with sampled negatives?

    def __init__(self, n_users: int, n_items: int, seed: int = 0, **kw):
        self.n_users = n_users
        self.n_items = n_items
        self.seed = seed
        self._X = None

    def fit(self, data) -> "BaseModel":
        raise NotImplementedError

    def scores(self, user_idx: int) -> np.ndarray:
        raise NotImplementedError

    def scores_batch(self, user_indices) -> np.ndarray:
        """Default: stack per-user scores. Override for matrix-friendly models."""
        return np.vstack([self.scores(int(u)) for u in user_indices])


class MostPop(BaseModel):
    name = "MostPop"

    def fit(self, data):
        self._X = data.X
        self._pop = np.asarray(data.X.sum(axis=0)).ravel().astype(float)
        return self

    def scores(self, user_idx):
        return self._pop

    def scores_batch(self, user_indices):
        return np.tile(self._pop, (len(user_indices), 1))


class ItemKNN(BaseModel):
    name = "ItemKNN"

    def __init__(self, n_users, n_items, seed=0, topk=100, **kw):
        super().__init__(n_users, n_items, seed)
        self.topk = topk

    def fit(self, data):
        X = data.X.tocsr()
        self._X = X
        col_norm = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        col_norm[col_norm == 0] = 1.0
        Xn = X.multiply(1.0 / col_norm)
        S = (Xn.T @ Xn).tocsr()
        S.setdiag(0.0); S.eliminate_zeros()
        self._S = _topk_rows(S, self.topk)
        return self

    def scores(self, user_idx):
        return np.asarray((self._X.getrow(user_idx) @ self._S).todense()).ravel()

    def scores_batch(self, user_indices):
        return np.asarray((self._X[np.asarray(user_indices)] @ self._S).todense())


class BPRMFnp(BaseModel):
    """BPR matrix factorization (numpy SGD). Reference/CPU fallback."""
    name = "BPRMFnp"
    uses_train_neg = True

    def __init__(self, n_users, n_items, seed=0, dim=32, lr=0.05, reg=0.01,
                 epochs=15, neg_sampling="uniform", **kw):
        super().__init__(n_users, n_items, seed)
        self.dim, self.lr, self.reg, self.epochs = dim, lr, reg, epochs
        self.neg_sampling = neg_sampling

    def fit(self, data):
        X = data.X.tocsr(); self._X = X
        rng = np.random.default_rng(self.seed)
        self.P = rng.normal(0, 0.1, (self.n_users, self.dim))
        self.Q = rng.normal(0, 0.1, (self.n_items, self.dim))
        pop = np.asarray(X.sum(axis=0)).ravel().astype(float)
        pop_p = pop / pop.sum() if pop.sum() > 0 else None
        user_pos = {u: X.indices[X.indptr[u]:X.indptr[u + 1]] for u in range(self.n_users)}
        active = [u for u in range(self.n_users) if len(user_pos[u]) > 0]
        if not active:
            return self                      # no training data -> keep random init
        for _ in range(self.epochs):
            us = rng.choice(active, size=max(1, X.nnz))
            if self.neg_sampling == "popularity" and pop_p is not None:
                js = rng.choice(self.n_items, size=len(us), p=pop_p)
            else:
                js = rng.integers(self.n_items, size=len(us))
            for u, j in zip(us, js):
                pos = user_pos[u]
                i = pos[rng.integers(len(pos))]
                pu, qi, qj = self.P[u], self.Q[i], self.Q[j]
                sig = 1.0 / (1.0 + np.exp(pu @ (qi - qj)))
                self.P[u] += self.lr * (sig * (qi - qj) - self.reg * pu)
                self.Q[i] += self.lr * (sig * pu - self.reg * qi)
                self.Q[j] += self.lr * (-sig * pu - self.reg * qj)
        return self

    def scores(self, user_idx):
        return self.P[user_idx] @ self.Q.T

    def scores_batch(self, user_indices):
        return self.P[np.asarray(user_indices)] @ self.Q.T


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BPRMF(BaseModel):
    """BPR matrix factorization (torch, mini-batch, auto CUDA->CPU)."""
    name = "BPRMF"
    uses_train_neg = True

    def __init__(self, n_users, n_items, seed=0, dim=64, lr=1e-2, reg=1e-5,
                 epochs=30, batch=4096, neg_sampling="uniform", **kw):
        super().__init__(n_users, n_items, seed)
        self.dim, self.lr, self.reg = dim, lr, reg
        self.epochs, self.batch, self.neg_sampling = epochs, batch, neg_sampling

    def fit(self, data):
        import torch
        X = data.X.tocoo(); self._X = data.X.tocsr()
        torch.manual_seed(self.seed); dev = _device()
        pu = torch.as_tensor(X.row, dtype=torch.long, device=dev)
        pi = torch.as_tensor(X.col, dtype=torch.long, device=dev)
        n_pos = pu.shape[0]
        if n_pos == 0:                       # no training interactions -> degenerate
            self.P = np.zeros((self.n_users, self.dim))
            self.Q = np.zeros((self.n_items, self.dim))
            return self
        P = torch.nn.Embedding(self.n_users, self.dim).to(dev)
        Q = torch.nn.Embedding(self.n_items, self.dim).to(dev)
        torch.nn.init.normal_(P.weight, std=0.1); torch.nn.init.normal_(Q.weight, std=0.1)
        opt = torch.optim.Adam([P.weight, Q.weight], lr=self.lr)
        g = torch.Generator(device=dev).manual_seed(self.seed)
        pop = torch.as_tensor(np.asarray(self._X.sum(axis=0)).ravel(),
                              dtype=torch.float, device=dev) + 1e-6
        for _ in range(self.epochs):
            perm = torch.randperm(n_pos, device=dev, generator=g)
            for b in range(0, n_pos, self.batch):
                idx = perm[b:b + self.batch]
                u, i = pu[idx], pi[idx]
                if self.neg_sampling == "popularity":
                    j = torch.multinomial(pop, idx.shape[0], replacement=True, generator=g)
                else:
                    j = torch.randint(self.n_items, (idx.shape[0],), device=dev, generator=g)
                eu, ei, ej = P(u), Q(i), Q(j)
                x = (eu * (ei - ej)).sum(1)
                loss = -torch.nn.functional.logsigmoid(x).mean()
                loss = loss + self.reg * (eu.pow(2).sum() + ei.pow(2).sum() + ej.pow(2).sum()) / idx.shape[0]
                opt.zero_grad(); loss.backward(); opt.step()
        self.P = P.weight.detach().cpu().numpy(); self.Q = Q.weight.detach().cpu().numpy()
        return self

    def scores(self, user_idx):
        return self.P[user_idx] @ self.Q.T

    def scores_batch(self, user_indices):
        return self.P[np.asarray(user_indices)] @ self.Q.T


class LightGCN(BaseModel):
    """LightGCN (He et al. 2020), BPR loss, torch."""
    name = "LightGCN"
    uses_train_neg = True

    def __init__(self, n_users, n_items, seed=0, dim=64, n_layers=3, lr=1e-2,
                 reg=1e-4, epochs=20, batch=2048, neg_sampling="uniform", **kw):
        super().__init__(n_users, n_items, seed)
        self.dim, self.n_layers, self.lr = dim, n_layers, lr
        self.reg, self.epochs, self.batch = reg, epochs, batch
        self.neg_sampling = neg_sampling

    def _norm_adj(self, X):
        import torch
        n = self.n_users + self.n_items
        Xc = X.tocoo()
        ui = np.stack([Xc.row, Xc.col + self.n_users])
        iu = np.stack([Xc.col + self.n_users, Xc.row])
        idx = np.concatenate([ui, iu], axis=1)
        deg = np.zeros(n); np.add.at(deg, idx[0], 1.0)
        with np.errstate(divide="ignore"):
            dinv = np.power(deg, -0.5)
        dinv[~np.isfinite(dinv)] = 0.0
        vals = dinv[idx[0]] * dinv[idx[1]]
        i = torch.tensor(idx, dtype=torch.long); v = torch.tensor(vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(i, v, (n, n)).coalesce().to(self.dev)

    def fit(self, data):
        import torch
        self._X = data.X.tocsr()
        torch.manual_seed(self.seed); self.dev = _device()
        A = self._norm_adj(data.X)
        E = torch.nn.Parameter(torch.empty(self.n_users + self.n_items, self.dim, device=self.dev))
        torch.nn.init.normal_(E, std=0.1)
        opt = torch.optim.Adam([E], lr=self.lr)
        X = self._X
        user_pos = [X.indices[X.indptr[u]:X.indptr[u + 1]] for u in range(self.n_users)]
        active = np.array([u for u in range(self.n_users) if len(user_pos[u]) > 0])
        if active.size == 0 or X.nnz == 0:   # degenerate after filtering
            self.U = np.zeros((self.n_users, self.dim))
            self.I = np.zeros((self.n_items, self.dim))
            return self
        rng = np.random.default_rng(self.seed)
        pop = np.asarray(X.sum(axis=0)).ravel().astype(float)
        pop_p = pop / pop.sum() if pop.sum() > 0 else None

        def propagate(emb):
            outs = [emb]
            for _ in range(self.n_layers):
                emb = torch.sparse.mm(A, emb); outs.append(emb)
            return torch.stack(outs, 0).mean(0)

        n_batches = max(1, X.nnz // self.batch)
        for _ in range(self.epochs):
            for _ in range(n_batches):
                u = rng.choice(active, size=self.batch)
                i = np.array([user_pos[uu][rng.integers(len(user_pos[uu]))] for uu in u])
                if self.neg_sampling == "popularity" and pop_p is not None:
                    j = rng.choice(self.n_items, size=self.batch, p=pop_p)
                else:
                    j = rng.integers(self.n_items, size=self.batch)
                u_t = torch.as_tensor(u, device=self.dev)
                i_t = torch.as_tensor(i + self.n_users, device=self.dev)
                j_t = torch.as_tensor(j + self.n_users, device=self.dev)
                allE = propagate(E)
                eu, ei, ej = allE[u_t], allE[i_t], allE[j_t]
                loss = -torch.log(torch.sigmoid((eu * ei).sum(1) - (eu * ej).sum(1)) + 1e-9).mean()
                loss = loss + self.reg * (eu.pow(2).sum() + ei.pow(2).sum() + ej.pow(2).sum()) / self.batch
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            allE = propagate(E)
            self.U = allE[:self.n_users].cpu().numpy()
            self.I = allE[self.n_users:].cpu().numpy()
        return self

    def scores(self, user_idx):
        return self.U[user_idx] @ self.I.T

    def scores_batch(self, user_indices):
        return self.U[np.asarray(user_indices)] @ self.I.T


class EASE(BaseModel):
    """Embarrassingly Shallow Autoencoder (Steck, WWW 2019). Dense closed-form
    item-item weights; single hyperparameter reg (lambda).

    The dense Gram matrix is O(n_items^2); on very large catalogues (e.g.
    Gowalla) this exceeds memory, so the runner/driver skips EASE there."""
    name = "EASE"
    uses_train_neg = False

    def __init__(self, n_users, n_items, seed=0, reg=250.0, **kw):
        super().__init__(n_users, n_items, seed)
        self.reg = float(reg)

    def fit(self, data):
        X = data.X.tocsr().astype(np.float64)
        self._X = X
        G = (X.T @ X).toarray()
        d = np.diag_indices(G.shape[0])
        G[d] += self.reg
        P = np.linalg.inv(G)
        B = -P / np.diag(P)
        B[d] = 0.0
        self._B = B
        return self

    def scores(self, user_idx):
        return np.asarray(self._X.getrow(user_idx) @ self._B).ravel()

    def scores_batch(self, user_indices):
        return np.asarray(self._X[np.asarray(user_indices)] @ self._B)


class SASRec(BaseModel):
    """SASRec (Kang & McAuley 2018). Sequential; uses its own next-item objective.

    Paper configuration (§4.2): 2 attention heads, batch 128, up to 100 epochs
    with early stopping (patience 10 on the training loss), tuned over
    lr∈{1e-3,5e-4}, maxlen∈{50,200}, dropout∈{0.2,0.5}. This is the strengthened
    SASRec whose scores back every SASRec number in the manuscript and in
    results_all_full.csv. (The small batch matters on small catalogues: a large
    batch gives too few updates per epoch and badly under-trains SASRec.)"""
    name = "SASRec"

    def __init__(self, n_users, n_items, seed=0, dim=64, n_blocks=2, n_heads=2,
                 maxlen=50, lr=1e-3, epochs=100, batch=128, dropout=0.2,
                 max_patience=10, **kw):
        super().__init__(n_users, n_items, seed)
        self.dim, self.n_blocks, self.n_heads = dim, n_blocks, n_heads
        self.maxlen, self.lr, self.epochs = maxlen, lr, epochs
        self.batch, self.dropout = batch, dropout
        self.max_patience = max_patience

    def _build_net(self):
        import torch.nn as nn
        pad = self.n_items
        net = nn.Module()
        net.item_emb = nn.Embedding(self.n_items + 1, self.dim, padding_idx=pad)
        net.pos_emb = nn.Embedding(self.maxlen, self.dim)
        net.drop = nn.Dropout(self.dropout)
        net.blocks = nn.ModuleList([nn.ModuleDict({
            "attn": nn.MultiheadAttention(self.dim, self.n_heads, dropout=self.dropout, batch_first=True),
            "ln1": nn.LayerNorm(self.dim), "ln2": nn.LayerNorm(self.dim),
            "ff1": nn.Linear(self.dim, self.dim), "ff2": nn.Linear(self.dim, self.dim),
        }) for _ in range(self.n_blocks)])
        net.ln_final = nn.LayerNorm(self.dim)
        return net, pad

    def fit(self, data):
        import torch
        import torch.nn.functional as F
        self._X = data.X.tocsr()
        torch.manual_seed(self.seed); self.dev = _device()
        seqs = (data.train_df.sort_values("ts").groupby("user")["item"].apply(list).to_dict())
        users = [u for u, s in seqs.items() if len(s) >= 2]
        self.net, self.pad = self._build_net(); net = self.net.to(self.dev)
        if not users:                         # no sequences of length >= 2
            self._item_mat = net.item_emb.weight[:self.n_items].detach().cpu().numpy()
            self._user_vec = {}
            self._fallback = self._item_mat.mean(0) if self.n_items else np.zeros(self.dim)
            return self
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        rng = np.random.default_rng(self.seed); L = self.maxlen

        def make_batch(bu):
            seq = np.full((len(bu), L), self.pad, np.int64)
            pos = np.full((len(bu), L), self.pad, np.int64)
            neg = np.full((len(bu), L), self.pad, np.int64)
            for bi, u in enumerate(bu):
                items = seqs[u][-(L + 1):]; inp, tgt = items[:-1], items[1:]
                seq[bi, L - len(inp):] = inp; pos[bi, L - len(tgt):] = tgt
                for t in range(L - len(tgt), L):
                    neg[bi, t] = rng.integers(self.n_items)
            return (torch.as_tensor(seq, device=self.dev),
                    torch.as_tensor(pos, device=self.dev),
                    torch.as_tensor(neg, device=self.dev))

        def encode(seq):
            mask = (seq != self.pad).unsqueeze(-1).float()
            positions = torch.arange(L, device=self.dev).unsqueeze(0).expand_as(seq)
            x = net.item_emb(seq) + net.pos_emb(positions)
            x = net.drop(x) * mask
            am = torch.triu(torch.ones(L, L, device=self.dev), diagonal=1).bool()
            for blk in net.blocks:
                q = blk["ln1"](x)
                a, _ = blk["attn"](q, x, x, attn_mask=am, need_weights=False)
                x = x + a
                x = x + blk["ff2"](F.relu(blk["ff1"](blk["ln2"](x))))
                x = x * mask
            return net.ln_final(x)

        # training with early stopping (epoch cap = self.epochs, e.g. 100)
        best_loss = float("inf"); patience = 0
        for _ in range(self.epochs):
            rng.shuffle(users)
            epoch_loss = 0.0; nbat = 0
            for b in range(0, len(users), self.batch):
                bu = users[b:b + self.batch]
                if not bu: continue
                seq, pos, neg = make_batch(bu)
                feat = encode(seq)
                pe, ne = net.item_emb(pos), net.item_emb(neg)
                valid = (pos != self.pad).float()
                pl = (feat * pe).sum(-1); nl = (feat * ne).sum(-1)
                loss = -(torch.log(torch.sigmoid(pl) + 1e-9) * valid).sum() / valid.sum()
                loss = loss - (torch.log(1 - torch.sigmoid(nl) + 1e-9) * valid).sum() / valid.sum()
                opt.zero_grad(); loss.backward(); opt.step()
                epoch_loss += float(loss.detach()); nbat += 1
            epoch_loss /= max(nbat, 1)
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss; patience = 0
            else:
                patience += 1
                if patience >= self.max_patience:
                    break

        net.eval()
        with torch.no_grad():
            self._item_mat = net.item_emb.weight[:self.n_items].detach().cpu().numpy()
            self._user_vec = {}
            for u in users:
                items = seqs[u][-L:]
                s = np.full((1, L), self.pad, np.int64); s[0, L - len(items):] = items
                feat = encode(torch.as_tensor(s, device=self.dev))
                self._user_vec[u] = feat[0, -1].detach().cpu().numpy()
        self._fallback = self._item_mat.mean(0)
        return self

    def scores(self, user_idx):
        v = self._user_vec.get(user_idx)
        return self._item_mat @ (self._fallback if v is None else v)

    def scores_batch(self, user_indices):
        V = np.vstack([self._user_vec.get(int(u), self._fallback) for u in user_indices])
        return V @ self._item_mat.T


def _topk_rows(S, k):
    S = S.tocsr(); data, indices, indptr = [], [], [0]
    for r in range(S.shape[0]):
        st, en = S.indptr[r], S.indptr[r + 1]
        rd, ri = S.data[st:en], S.indices[st:en]
        if len(rd) > k:
            top = np.argpartition(rd, -k)[-k:]; rd, ri = rd[top], ri[top]
        data.append(rd); indices.append(ri); indptr.append(indptr[-1] + len(rd))
    return sp.csr_matrix((np.concatenate(data) if data else np.array([]),
                          np.concatenate(indices) if indices else np.array([], int),
                          np.array(indptr)), shape=S.shape)


MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "MostPop": MostPop, "ItemKNN": ItemKNN, "EASE": EASE, "BPRMF": BPRMF,
    "BPRMFnp": BPRMFnp, "LightGCN": LightGCN, "SASRec": SASRec,
}
