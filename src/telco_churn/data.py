"""Chargement, contrôle d'intégrité et nettoyage."""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger(__name__)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: Path = config.DATA_RAW, verify: bool = True) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} introuvable. Voir data/README.md pour télécharger le dataset.")
    if verify and sha256_of(path) != config.DATA_SHA256:
        log.warning("Empreinte SHA-256 inattendue : le fichier a-t-il été modifié ?")
    return pd.read_csv(path, dtype={config.ID_COL: str})


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Règles de nettoyage documentées. Retourne aussi un journal des corrections."""
    report = {"rows_in": len(df)}
    df = df.copy()
    report["duplicate_ids"] = int(df[config.ID_COL].duplicated().sum())
    df = df.drop_duplicates(config.ID_COL)

    # TotalCharges est un texte ; il est vide pour les clients arrivés ce mois-ci (tenure = 0) :
    # ils n'ont encore rien payé, on met 0.
    total = pd.to_numeric(df["TotalCharges"].astype(str).str.strip(), errors="coerce")
    blank = total.isna()
    report["total_charges_blank_set_to_0"] = int(blank.sum())
    report["blank_rows_all_tenure_0"] = bool((df.loc[blank, "tenure"] == 0).all())
    df["TotalCharges"] = total.fillna(0.0)

    df[config.TARGET] = df[config.TARGET].map({"Yes": 1, "No": 0}).astype(int)
    report["rows_out"] = len(df)
    report["churn_rate"] = round(float(df[config.TARGET].mean()), 4)
    return df.reset_index(drop=True), report


def split_xy(df: pd.DataFrame):
    return df[config.FEATURES], df[config.TARGET].astype(int)
