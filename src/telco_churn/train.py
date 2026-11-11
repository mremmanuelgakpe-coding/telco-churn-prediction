"""Entraînement, comparaison des 9 modèles, choix du seuil de campagne et export.

Usage : PYTHONPATH=src python -m telco_churn.train
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split

from . import __version__, config, plots
from .data import clean, load_raw, sha256_of, split_xy
from .explain import attach_background, shap_values
from .metrics import best_threshold, decile_table, decision_metrics, score_metrics
from .models import EXPLAINABLE, build_pipeline, candidate_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


def run() -> dict:
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df, cleaning = clean(load_raw())
    log.info("Nettoyage : %s", cleaning)
    X, y = split_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=config.TEST_SIZE, stratify=y,
                                              random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    # 1) Comparaison en validation croisée ; le test ne sert pas au choix
    rows, oof, test_p, fitted = [], {}, {}, {}
    for name, model in candidate_models().items():
        t0 = time.perf_counter()
        pipe = build_pipeline(model)
        p_oof = cross_val_predict(pipe, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
        folds = [score_metrics(y_tr.iloc[va], p_oof[va])["roc_auc"] for _, va in cv.split(X_tr, y_tr)]
        m = score_metrics(y_tr, p_oof)
        pipe.fit(X_tr, y_tr)
        p_te = pipe.predict_proba(X_te)[:, 1]
        rows.append({"model": name, "cv_roc_auc": m["roc_auc"], "cv_roc_auc_std": round(float(np.std(folds)), 4),
                     "cv_pr_auc": m["pr_auc"], "cv_brier": m["brier"], "cv_lift_top20": m["lift_top20"],
                     "test_roc_auc": score_metrics(y_te, p_te)["roc_auc"], "explainable": name in EXPLAINABLE,
                     "train_seconds": round(time.perf_counter() - t0, 1)})
        oof[name], test_p[name], fitted[name] = p_oof, p_te, pipe
        log.info("%-20s CV AUC=%.4f", name, m["roc_auc"])

    comp = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    comp.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    best_overall = comp.iloc[0]["model"]
    selected = comp[comp["explainable"]].iloc[0]["model"]
    log.info("Meilleur global : %s ; modèle retenu (explicable) : %s", best_overall, selected)

    # 2) Seuil de ciblage : maximise la valeur de la campagne sur les prédictions out-of-fold
    thr, curve = best_threshold(y_tr, oof[selected], X_tr["MonthlyCharges"])
    plots.profit_curve(curve, thr, config.FIGURES_DIR / "campaign_value_threshold.png")

    # 3) Évaluation finale sur le test, une seule fois
    model = fitted[selected]
    attach_background(model, X_tr)
    p = test_p[selected]
    test_scores = score_metrics(y_te, p)
    decision = decision_metrics(y_te, p, X_te["MonthlyCharges"], thr)
    deciles = decile_table(y_te, p)
    deciles.to_csv(config.REPORTS_DIR / "deciles.csv", index=False)

    # Table agrégée par seuil (jeu de test) : permet de recalculer la valeur de campagne
    # pour n'importe quelles hypothèses, sans publier de données client individuelles.
    yv, mv = y_te.to_numpy(), X_te["MonthlyCharges"].to_numpy()
    camp = []
    for t in np.round(np.arange(0.0, 1.001, 0.01), 2):
        tg = p >= t
        camp.append({"threshold": float(t), "targeted": int(tg.sum()), "churners_targeted": int(yv[tg].sum()),
                     "monthly_targeted": round(float(mv[tg].sum()), 2),
                     "monthly_churners_targeted": round(float((yv * mv)[tg].sum()), 2)})
    pd.DataFrame(camp).to_csv(config.REPORTS_DIR / "campaign_table_test.csv", index=False)

    sv, base = shap_values(model, X_te, config.CATEGORICAL)
    mean_abs = sv.abs().mean().sort_values(ascending=False)
    mean_abs.round(4).rename("mean_abs_shap").to_csv(config.REPORTS_DIR / "shap_global.csv")

    # 4) Figures
    fd = config.FIGURES_DIR
    plots.comparison_bar(comp, selected, fd / "model_comparison.png")
    shown = [selected] + [m for m in comp["model"] if m != selected][:3]
    plots.roc_pr(y_te, {m: test_p[m] for m in shown}, fd / "roc_pr.png")
    plots.calibration(y_te, p, fd / "calibration.png")
    plots.shap_global(list(sv.columns), sv.abs().mean().to_numpy(), fd / "shap_global.png")
    i = int(np.argmin(np.abs(p - 0.75)))
    plots.shap_example(X_te.iloc[i].to_dict(), sv.iloc[i].to_dict(), float(1 / (1 + np.exp(-base[i]))), float(p[i]),
                       fd / "shap_example.png")
    plots.churn_by(df, "Contract", config.TARGET, fd / "eda_churn_by_contract.png", "Taux de churn par type de contrat",
                   ["Month-to-month", "One year", "Two year"])
    bands = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
    plots.churn_by(df.assign(tenure_band=bands), "tenure_band", config.TARGET, fd / "eda_churn_by_tenure.png",
                   "Taux de churn par ancienneté (mois)")

    # 5) Export de l'artefact et des métadonnées
    path = config.MODELS_DIR / "model.joblib"
    joblib.dump(model, path)
    meta = {
        "project": "telco-churn-prediction", "code_version": __version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": selected, "best_overall_model": best_overall, "features": config.FEATURES,
        "numeric_features": config.NUMERIC + config.BINARY, "categorical_features": config.CATEGORICAL,
        "categories": {c: sorted(df[c].unique().tolist()) for c in config.CATEGORICAL},
        "threshold": thr,
        "campaign": {"offer_months": config.OFFER_MONTHS, "success_rate": config.SUCCESS_RATE,
                     "retained_months": config.RETAINED_MONTHS},
        "data_sha256": sha256_of(config.DATA_RAW), "model_sha256": sha256_of(path),
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "test_metrics": test_scores,
    }
    (config.MODELS_DIR / "model.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    report = {
        "cleaning": cleaning, "selected_model": selected, "best_overall_model": best_overall,
        "comparison": comp.to_dict(orient="records"), "test_metrics": test_scores, "test_decision": decision,
        "deciles": deciles.to_dict(orient="records"), "shap_mean_abs": mean_abs.round(4).to_dict(),
        "campaign_curve_oof": [{"threshold": t, "net_value": round(v, 1), "targeted": n} for t, v, n in curve],
        "churn_by_contract": df.groupby("Contract")[config.TARGET].mean().round(4).to_dict(),
        "test_size": {"customers": int(len(y_te)), "churners": int(y_te.sum())},
    }
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log.info("Test : %s | décision : %s", test_scores, decision)
    return report


if __name__ == "__main__":
    run()
