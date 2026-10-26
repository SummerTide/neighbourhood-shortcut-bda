"""Building-level metrics for four-level damage assessment.

Scores are pooled over all evaluated buildings, as the xView2 metric pools over
the test set; per-disaster macro-F1 is reported alongside. "hmean_f1" is the
harmonic mean of the four per-class F1 scores (the form of the xView2 damage
score) computed per building rather than per pixel.
"""
import os

import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score

N_CLASSES = int(os.environ.get("BDA_N_CLASSES", 4))  # 3 for DisasterM3 (intact / damaged / destroyed)


def _scores(y, p):
    pred = p.argmax(1)
    f1 = f1_score(y, pred, labels=range(N_CLASSES), average=None, zero_division=0)
    out = {f"f1_{c}": float(f1[c]) for c in range(N_CLASSES)}
    out["macro_f1"] = float(f1.mean())
    out["hmean_f1"] = float(N_CLASSES / np.sum(1.0 / np.clip(f1, 1e-6, None)))
    out["qwk"] = float(cohen_kappa_score(y, pred, weights="quadratic"))
    out["acc"] = float((pred == y).mean())
    out["bal_acc"] = float(balanced_accuracy_score(y, pred))
    present = np.unique(y)
    if len(present) == N_CLASSES:
        pe = np.clip(p, 1e-9, None)
        out["auc_ovr"] = float(roc_auc_score(y, pe / pe.sum(1, keepdims=True), multi_class="ovr", labels=range(N_CLASSES)))
    if 0 < (y > 0).mean() < 1:
        out["auc_bin"] = float(roc_auc_score((y > 0).astype(int), 1 - p[:, 0]))
    return out


def pred_metrics(y, pred):
    """Scores from hard predictions where -1 marks a building without a prediction
    (counted as wrong for F1; QWK over predicted buildings only)."""
    f1 = f1_score(y, pred, labels=range(N_CLASSES), average=None, zero_division=0)
    out = {f"f1_{c}": float(f1[c]) for c in range(N_CLASSES)}
    out["macro_f1"] = float(f1.mean())
    out["hmean_f1"] = float(N_CLASSES / np.sum(1.0 / np.clip(f1, 1e-6, None)))
    has = pred >= 0
    out["coverage"] = float(has.mean())
    out["qwk"] = float(cohen_kappa_score(y[has], pred[has], weights="quadratic")) if has.sum() > 1 else float("nan")
    out["acc"] = float((pred == y).mean())
    return out


def building_metrics(y, p, disasters=None):
    res = {"overall": _scores(y, p)}
    if disasters is not None:
        res["per_disaster"] = {}
        for d in sorted(set(disasters)):
            m = disasters == d
            if m.sum() > 0:
                f1 = f1_score(y[m], p[m].argmax(1), labels=range(N_CLASSES), average=None, zero_division=0)
                res["per_disaster"][d] = float(f1.mean())
    return res
