import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix


def threshold_at_fpr(y, score, target):
    """Largest-recall threshold meeting target FPR; +inf means never alert."""
    candidates = np.r_[np.inf, np.unique(score)[::-1]]
    best, best_recall = np.inf, -1.0
    for threshold in candidates:
        pred = score >= threshold
        neg = (y == 0).sum()
        fpr = ((pred == 1) & (y == 0)).sum() / neg
        recall = ((pred == 1) & (y == 1)).sum() / max(1, (y == 1).sum())
        # Scan every candidate: maximize recall, and use the more conservative
        # (higher) threshold when recall is tied.
        if fpr <= target + 1e-12 and (recall > best_recall or (recall == best_recall and threshold > best)):
            best, best_recall = threshold, recall
    return float(best)


def metrics(y, score, threshold):
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    precision = tp/(tp+fp) if tp+fp else np.nan
    return dict(recall=tp/(tp+fn) if tp+fn else 0.0, precision=precision,
                fpr=fp/(fp+tn) if fp+tn else 0.0, ap=average_precision_score(y, score),
                alerts=int(pred.sum()), false_alerts_per_1000=1000*fp/(fp+tn) if fp+tn else 0.0,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def paired_group_bootstrap(y, score_a, threshold_a, score_b, threshold_b, groups, rng, n=2000):
    unique = np.unique(groups); deltas = []
    for _ in range(n):
        selected = rng.choice(unique, len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(groups == g) for g in selected])
        ma = metrics(y[idx], score_a[idx], threshold_a)["recall"]
        mb = metrics(y[idx], score_b[idx], threshold_b)["recall"]
        deltas.append(ma-mb)
    return np.percentile(deltas, [2.5, 97.5])
