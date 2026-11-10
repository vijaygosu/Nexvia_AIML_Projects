"""Account/device/email graph, structural node features, GNN (PyTorch Geometric)
and a label-based connected-component 'ring' baseline that needs no torch."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ENTITY_COLS = ("device_id", "email_domain")
UNKNOWN = -1


def build_graph(df: pd.DataFrame, max_degree: int = 50):
    """Bipartite account<->entity graph.

    Entities linked to more than `max_degree` accounts (gmail.com, a mall's wifi
    device...) are dropped: they connect everyone and carry no ring signal.
    Returns (users Index, edges [u_node, entity_node], n_nodes).
    """
    users = pd.Index(df["user_id"].unique())
    uidx = users.get_indexer(df["user_id"])
    offset, parts = len(users), []
    for col in ENTITY_COLS:
        vals = df[col].to_numpy()
        m = vals != UNKNOWN
        e = pd.DataFrame({"u": uidx[m], "e": vals[m]}).drop_duplicates()
        e = e[e.groupby("e")["u"].transform("size") <= max_degree]
        codes, uniq = pd.factorize(e["e"])
        parts.append(np.column_stack([e["u"].to_numpy(), codes + offset]))
        offset += len(uniq)
    edges = np.vstack(parts) if parts else np.zeros((0, 2), dtype=np.int64)
    return users, edges.astype(np.int64), offset


def _components(edges, n_nodes):
    a = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes))
    return connected_components(a, directed=False)


def user_labels(df: pd.DataFrame, users: pd.Index, t_end: float):
    """Labels/mask using ONLY the training window (ts < t_end): no future labels."""
    tr = df[df["ts"] < t_end]
    pos = pd.Index(tr.loc[tr["is_fraud"] == 1, "user_id"].unique())
    mask = users.isin(tr["user_id"].unique())
    y = users.isin(pos).astype(int)
    return y, mask


def ring_scores(users, edges, n_nodes, y, mask) -> pd.Series:
    """Baseline: #OTHER label-known fraud accounts in the same connected component."""
    n_u = len(users)
    _, lab = _components(edges, n_nodes)
    bad = (np.asarray(y) == 1) & np.asarray(mask)
    comp_bad = np.bincount(lab[:n_u][bad], minlength=lab.max() + 1)
    k = comp_bad[lab[:n_u]] - bad.astype(int)
    return pd.Series(1 - np.exp(-0.7 * k), index=users, name="gnn_score")


def node_features(df, users, edges, n_nodes, t_end) -> np.ndarray:
    """Structural features + train-window account stats (no test-window content)."""
    n_u = len(users)
    deg = np.bincount(edges.ravel(), minlength=n_nodes).astype(float)
    nb = (np.bincount(edges[:, 0], weights=deg[edges[:, 1]], minlength=n_nodes)
          + np.bincount(edges[:, 1], weights=deg[edges[:, 0]], minlength=n_nodes))
    nb_mean = nb / np.maximum(deg, 1)
    ncomp, lab = _components(edges, n_nodes)
    comp_users = np.bincount(lab[:n_u], minlength=ncomp)
    comp_size = comp_users[lab].astype(float)
    tr = df[df["ts"] < t_end]
    g = tr.groupby("user_id")["amount"].agg(["size", lambda s: np.log1p(s).mean()])
    g.columns = ["n", "mlog"]
    g = g.reindex(users).fillna(0.0)
    n_tr = np.zeros(n_nodes)
    m_tr = np.zeros(n_nodes)
    n_tr[:n_u], m_tr[:n_u] = np.log1p(g["n"].to_numpy()), g["mlog"].to_numpy()
    is_user = np.zeros(n_nodes)
    is_user[:n_u] = 1
    return np.column_stack([np.log1p(deg), np.log1p(nb_mean), np.log1p(comp_size), n_tr, m_tr, is_user])


def gnn_scores(users, edges, n_nodes, X, y, mask, epochs=200, hidden=64, seed=0, lr=0.01) -> pd.Series:
    """GraphSAGE node classifier (PyTorch Geometric). Trained on train-window labels only."""
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv

    torch.manual_seed(seed)
    n_u = len(users)
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    x = torch.tensor(Xs, dtype=torch.float)
    ei = torch.tensor(np.vstack([edges, edges[:, ::-1]]).T.copy(), dtype=torch.long)
    yt = torch.tensor(np.asarray(y), dtype=torch.float)
    tm = torch.tensor(np.asarray(mask), dtype=torch.bool)

    class SAGE(torch.nn.Module):
        def __init__(s, d, h):
            super().__init__()
            s.c1, s.c2, s.out = SAGEConv(d, h), SAGEConv(h, h), torch.nn.Linear(h, 1)

        def forward(s, x, ei):
            x = F.dropout(F.relu(s.c1(x, ei)), 0.3, s.training)
            return s.out(F.relu(s.c2(x, ei))).squeeze(-1)

    model = SAGE(x.shape[1], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    pos = float(yt[tm].sum().clamp(min=1))
    pw = torch.tensor((float(tm.sum()) - pos) / pos)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        logits = model(x, ei)[:n_u]
        F.binary_cross_entropy_with_logits(logits[tm], yt[tm], pos_weight=pw).backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x, ei)[:n_u]).numpy()
    return pd.Series(p, index=users, name="gnn_score")


def compute_user_scores(df, t_train_end, backend="auto", seed=0):
    """Return (Series user_id -> score, backend_used)."""
    users, edges, n_nodes = build_graph(df)
    y, mask = user_labels(df, users, t_train_end)
    if backend in ("auto", "pyg"):
        try:
            X = node_features(df, users, edges, n_nodes, t_train_end)
            return gnn_scores(users, edges, n_nodes, X, y, mask, seed=seed), "pyg"
        except ImportError:
            if backend == "pyg":
                raise
    return ring_scores(users, edges, n_nodes, y, mask), "ring"
