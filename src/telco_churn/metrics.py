"""Métriques de classement et économie de la campagne de rétention."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)

from . import config


def score_metrics(y, p) -> dict:
    y = np.asarray(y)
    order = np.argsort(-p)
    top = order[: max(1, int(round(0.2 * len(y))))]
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "lift_top20": round(float(y[top].mean() / y.mean()), 3),
        "capture_top20": round(float(y[top].sum() / y.sum()), 4),
    }


def campaign_value(y, p, monthly, threshold,
                   offer=config.OFFER_MONTHS, success=config.SUCCESS_RATE, retained=config.RETAINED_MONTHS):
    """Valeur nette attendue (en $, la devise des factures) d'une campagne ciblant p >= seuil.

    Coût : chaque client ciblé reçoit une offre valant ``offer`` mois de son abonnement.
    Gain : un vrai churner ciblé est retenu avec la probabilité ``success`` ; on conserve alors
    ``retained`` mois de revenus.
    """
    y, p, m = np.asarray(y), np.asarray(p), np.asarray(monthly)
    target = p >= threshold
    gain = (y * success * retained * m)[target].sum()
    cost = (offer * m)[target].sum()
    return float(gain - cost), int(target.sum())


def best_threshold(y, p, monthly) -> tuple[float, list]:
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    curve = [(t, *campaign_value(y, p, monthly, t)) for t in grid]
    best = max(curve, key=lambda r: r[1])
    return float(best[0]), curve


def decision_metrics(y, p, monthly, threshold) -> dict:
    y = np.asarray(y)
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    value, n_target = campaign_value(y, p, monthly, threshold)
    value_all, _ = campaign_value(y, np.ones_like(p), monthly, 0.0)
    return {
        "threshold": threshold,
        "targeted_share": round(float(pred.mean()), 4),
        "precision_churn": round(float(precision_score(y, pred)), 4),
        "recall_churn": round(float(recall_score(y, pred)), 4),
        "net_value": round(value, 2),
        "net_value_target_everyone": round(value_all, 2),
        "net_value_per_customer": round(value / len(y), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def decile_table(y, p) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": p})
    df["decile"] = pd.qcut(df["p"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    out = df.groupby("decile", observed=True).agg(n=("y", "size"), proba_moyenne=("p", "mean"),
                                                  taux_churn_observe=("y", "mean"))
    out["lift"] = out["taux_churn_observe"] / df["y"].mean()
    out["capture_cumulee_churners"] = out["taux_churn_observe"].mul(out["n"]).cumsum() / df["y"].sum()
    return out.round(4).reset_index()
