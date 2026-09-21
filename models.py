"""Models with a common predict_proba interface."""
import numpy as np
import torch
from torch import nn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


def logistic_model(C=1.0):
    return Pipeline([
        ("prepare", ColumnTransformer([("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), list(range(30))) ])),
        ("model", LogisticRegression(C=C, max_iter=3000, solver="liblinear")),
    ])


class TabMNet(nn.Module):
    """Parameter-efficient MLP ensemble: shared trunk, member-specific heads."""
    def __init__(self, n_features=30, width=128, depth=2, members=8, dropout=0.1):
        super().__init__()
        layers = []
        n = n_features
        for _ in range(depth):
            layers += [nn.Linear(n, width), nn.ReLU(), nn.Dropout(dropout)]
            n = width
        self.trunk = nn.Sequential(*layers)
        self.member_heads = nn.Parameter(torch.empty(members, n, 1))
        self.member_bias = nn.Parameter(torch.zeros(members, 1))
        nn.init.xavier_uniform_(self.member_heads)

    def forward(self, x):
        h = self.trunk(x)
        return torch.einsum("bh,mho->bmo", h, self.member_heads).squeeze(-1) + self.member_bias.T


class TabMClassifier:
    def __init__(self, width=128, depth=2, lr=1e-3, weight_decay=1e-4, dropout=0.1,
                 members=8, seed=0, epochs=5, patience=2):
        # CPU is deliberately used here: small batches on MPS are slower and less reproducible.
        self.__dict__.update(locals()); self.device = "cpu"

    def fit(self, X, y, X_val, y_val):
        torch.manual_seed(self.seed)
        self.median_ = np.nanmedian(np.asarray(X, dtype=np.float32), axis=0)
        def prep(a):
            a = np.asarray(a, dtype=np.float32).copy()
            a[np.isnan(a)] = np.take(self.median_, np.where(np.isnan(a))[1])
            return a
        self.model = TabMNet(n_features=np.asarray(X).shape[1], width=self.width, depth=self.depth,
                             members=self.members, dropout=self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        xt, yt = torch.tensor(prep(X)), torch.tensor(y, dtype=torch.float32)
        xv, yv = torch.tensor(prep(X_val)), torch.tensor(y_val, dtype=torch.float32)
        best, best_state, stale = float("inf"), None, 0
        for _ in range(self.epochs):
            self.model.train(); order = torch.randperm(len(xt))
            for inds in order.split(256):
                logits = self.model(xt[inds].to(self.device)); target = yt[inds].to(self.device)[:, None].expand_as(logits)
                opt.zero_grad(); loss_fn(logits, target).backward(); opt.step()
            self.model.eval()
            with torch.no_grad():
                v = loss_fn(self.model(xv.to(self.device)), yv.to(self.device)[:, None].expand(-1, self.members)).item()
            if v < best - 1e-5:
                best, stale = v, 0; best_state = {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience: break
        self.model.load_state_dict(best_state); return self

    def predict_proba(self, X):
        a = np.asarray(X, dtype=np.float32).copy(); a[np.isnan(a)] = np.take(self.median_, np.where(np.isnan(a))[1])
        self.model.eval()
        with torch.no_grad(): p = torch.sigmoid(self.model(torch.tensor(a).to(self.device))).mean(1).cpu().numpy()
        return np.c_[1-p, p]


def tabpfn_model(seed):
    # Keep third-party model cache inside a writable task-local location.
    import os
    os.environ.setdefault("SKB_DATA_DIRECTORY", "/private/tmp/sukciavimas_skrub")
    os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")
    from pathlib import Path
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(random_state=seed, model_path=Path.cwd() / ".tabpfn_models" / "tabpfn-v2-classifier-v2_default.ckpt")
