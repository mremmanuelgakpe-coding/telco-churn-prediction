"""Tables agrégées de la vue « Activité » (marketing, direction).

Produit, sans aucun identifiant ni ligne client :
- ``activity_cube.csv`` : un cube par segment (contrat × Internet × paiement × senior × ancienneté ×
  niveau de risque) avec des sommes (clients, départs, revenu mensuel, services…), k-anonyme (k = 5).
  Tous les KPI du tableau de bord se recalculent par addition, quel que soit le filtre.
- ``tenure_curve.csv`` : clients et départs par mois d'ancienneté (0 à 72), par contrat et accès Internet.

Le niveau de risque vient du modèle entraîné, appliqué à toute la base : pour 80 % des clients ce sont
des scores « vus à l'entraînement » (usage de pilotage, pas de mesure de performance).

Usage : PYTHONPATH=src python -m telco_churn.activity
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from . import config
from .data import clean, load_raw

OPTIONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
TENURE_BANDS = ([-1, 6, 12, 24, 48, 72], ["0-6 mois", "7-12 mois", "13-24 mois", "25-48 mois", "49-72 mois"])
K_ANON = 5
GROUPED = "Regroupé"
RISK_BANDS = ([-0.01, 0.15, 0.35, 0.60, 1.0], ["Faible", "Modéré", "Élevé", "Très élevé"])


def build() -> dict:
    df, _ = clean(load_raw())
    model = joblib.load(config.MODELS_DIR / "model.joblib")
    meta = json.loads((config.MODELS_DIR / "model.meta.json").read_text())
    p = model.predict_proba(df[config.FEATURES])[:, 1]

    yes = lambda c: df[c].eq("Yes").astype(int)  # noqa: E731
    d = pd.DataFrame({
        "contract": df["Contract"], "internet": df["InternetService"], "payment": df["PaymentMethod"],
        "senior": df["SeniorCitizen"].map({1: "Senior", 0: "Non senior"}),
        "tenure_band": pd.cut(df["tenure"], TENURE_BANDS[0], labels=TENURE_BANDS[1]).astype(str),
        "risk_band": pd.cut(p, RISK_BANDS[0], labels=RISK_BANDS[1]).astype(str),
        "customers": 1, "options_sum": sum(yes(c) for c in OPTIONS), "churned": df[config.TARGET],
        "mrr": df["MonthlyCharges"], "mrr_churned": df["MonthlyCharges"] * df[config.TARGET],
        "total_charges": df["TotalCharges"], "tenure_sum": df["tenure"],
        "to_target": (p >= meta["threshold"]).astype(int),
        "mrr_to_target": df["MonthlyCharges"] * (p >= meta["threshold"]),
        "to_target_active": ((p >= meta["threshold"]) & df[config.TARGET].eq(0)).astype(int),
        "mrr_to_target_active": df["MonthlyCharges"] * ((p >= meta["threshold"]) & df[config.TARGET].eq(0)),
        "score_sum": p,
        "phone": yes("PhoneService"), "multiple_lines": yes("MultipleLines"),
        "fiber": df["InternetService"].eq("Fiber optic").astype(int),
        "dsl": df["InternetService"].eq("DSL").astype(int),
        "paperless": yes("PaperlessBilling"), "partner": yes("Partner"), "dependents": yes("Dependents"),
        **{f"opt_{c}": yes(c) for c in OPTIONS},
        **{f"churn_opt_{c}": yes(c) * df[config.TARGET] for c in OPTIONS},
    })
    dims = ["contract", "internet", "payment", "senior", "tenure_band", "risk_band"]
    cube = d.groupby(dims, observed=True).sum(numeric_only=True).reset_index()
    # k-anonymat : les cellules de moins de K clients sont fusionnées par type de contrat (dimensions
    # fines remplacées par « Regroupé »). Les totaux restent exacts ; aucune cellule publiée < K clients.
    small = cube["customers"] < K_ANON
    grouped = (cube[small].assign(**{c: GROUPED for c in dims[1:]})
               .groupby(dims, observed=True).sum(numeric_only=True).reset_index())
    cube = pd.concat([cube[~small], grouped], ignore_index=True)
    assert (cube["customers"] >= K_ANON).all(), "cellule regroupée encore trop petite"
    cube["mrr"] = cube["mrr"].round(2)
    for c in ("mrr_churned", "total_charges", "mrr_to_target", "mrr_to_target_active", "score_sum"):
        cube[c] = cube[c].round(2)

    curve = (pd.DataFrame({"contract": df["Contract"], "internet": df["InternetService"], "tenure": df["tenure"],
                           "customers": 1, "churned": df[config.TARGET]})
             .groupby(["contract", "internet", "tenure"]).sum().reset_index())

    out = config.REPORTS_DIR / "activity"
    out.mkdir(parents=True, exist_ok=True)
    cube.to_csv(out / "activity_cube.csv", index=False)
    curve.to_csv(out / "tenure_curve.csv", index=False)
    curve = curve[curve["customers"] >= K_ANON]  # mois d'ancienneté trop petits retirés de la courbe
    check = {"customers": int(cube["customers"].sum()), "churned": int(cube["churned"].sum()),
             "mrr": round(float(cube["mrr"].sum()), 2), "cells": int(len(cube)), "threshold": meta["threshold"],
             "k_anonymity": K_ANON, "customers_in_grouped_cells": int(cube.loc[cube["internet"] == GROUPED, "customers"].sum()),
             "min_cell_size": int(cube["customers"].min())}
    assert check["customers"] == len(df) and check["churned"] == int(df[config.TARGET].sum())
    (out / "manifest.json").write_text(json.dumps(check, indent=2, ensure_ascii=False))
    return check


if __name__ == "__main__":
    print(build())
