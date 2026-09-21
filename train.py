"""Leakage-safe selection of model hyperparameters from grouped OOF predictions."""
import numpy as np
from sklearn.metrics import average_precision_score
from models import logistic_model, TabMClassifier, tabpfn_model
from evaluate import threshold_at_fpr


def tune_logistic(X, y, splits):
    best_c, best = None, -np.inf
    for c in [0.01, 0.1, 1, 10, 100]:
        p = np.zeros(len(y))
        for tr, va in splits:
            m = logistic_model(c).fit(X.iloc[tr], y[tr]); p[va] = m.predict_proba(X.iloc[va])[:,1]
        val = average_precision_score(y, p)
        if val > best: best, best_c = val, c
    return best_c


def tune_tabm(X, y, splits, seed):
    configs = [dict(width=128,depth=2,lr=1e-3,weight_decay=1e-4,dropout=.1)]
    best_cfg, best = None, -np.inf
    # Fixed compact holdout inside each training fold supplies early stopping.
    for ci, cfg in enumerate(configs):
        p = np.zeros(len(y))
        for fold, (tr, va) in enumerate(splits):
            cut = max(1, int(.15*len(tr))); early, fit = tr[:cut], tr[cut:]
            m = TabMClassifier(**cfg, seed=seed + ci*10 + fold).fit(X.iloc[fit], y[fit], X.iloc[early], y[early])
            p[va] = m.predict_proba(X.iloc[va])[:,1]
        value = average_precision_score(y, p)
        if value > best: best, best_cfg = value, cfg
    return best_cfg


def oof_scores(kind, X, y, splits, seed, config=None):
    p = np.zeros(len(y))
    for fold, (tr, va) in enumerate(splits):
        if kind == "logistic": m = logistic_model(config).fit(X.iloc[tr], y[tr])
        elif kind == "tabm":
            cut=max(1,int(.15*len(tr))); m=TabMClassifier(**config, seed=seed+fold).fit(X.iloc[tr[cut:]],y[tr[cut:]],X.iloc[tr[:cut]],y[tr[:cut]])
        elif kind == "tabpfn": m = tabpfn_model(seed+fold).fit(X.iloc[tr], y[tr])
        p[va] = m.predict_proba(X.iloc[va])[:,1]
    return p
