"""API de prédiction du churn (FastAPI).

Sécurité : clé API (en-tête X-API-Key, comparaison à temps constant) ; validation stricte des entrées
(valeurs autorisées, bornes, champs inconnus refusés) ; contrôle SHA-256 du modèle avant chargement
(joblib = pickle) ; aucune donnée client journalisée ; documentation désactivable ; lots plafonnés.
Chaque prédiction est accompagnée de ses motifs (valeurs SHAP exactes).
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, ConfigDict, Field, model_validator

from telco_churn.explain import reason_codes, shap_values

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model.joblib"))
META_PATH = MODEL_PATH.with_suffix(".meta.json")
MAX_BATCH = int(os.getenv("MAX_BATCH", "500"))
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

YesNo = Literal["Yes", "No"]
NoPhone = Literal["Yes", "No", "No phone service"]
NoInternet = Literal["Yes", "No", "No internet service"]


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "tenure": 5, "MonthlyCharges": 89.5, "TotalCharges": 447.5, "SeniorCitizen": 0, "gender": "Female",
        "Partner": "No", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "Yes",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
        "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"}})

    tenure: int = Field(ge=0, le=120, description="Ancienneté en mois")
    MonthlyCharges: float = Field(gt=0, le=1000)
    TotalCharges: float = Field(ge=0, le=100_000)
    SeniorCitizen: Literal[0, 1]
    gender: Literal["Female", "Male"]
    Partner: YesNo
    Dependents: YesNo
    PhoneService: YesNo
    MultipleLines: NoPhone
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: NoInternet
    OnlineBackup: NoInternet
    DeviceProtection: NoInternet
    TechSupport: NoInternet
    StreamingTV: NoInternet
    StreamingMovies: NoInternet
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal["Electronic check", "Mailed check", "Bank transfer (automatic)",
                           "Credit card (automatic)"]

    @model_validator(mode="after")
    def coherent_services(self):
        if (self.PhoneService == "No") != (self.MultipleLines == "No phone service"):
            raise ValueError("MultipleLines doit valoir 'No phone service' si et seulement si PhoneService = 'No'")
        internet_opts = [self.OnlineSecurity, self.OnlineBackup, self.DeviceProtection, self.TechSupport,
                         self.StreamingTV, self.StreamingMovies]
        no_net = self.InternetService == "No"
        if any((o == "No internet service") != no_net for o in internet_opts):
            raise ValueError("Les options Internet valent 'No internet service' si et seulement si InternetService = 'No'")
        return self


class Factor(BaseModel):
    feature: str
    label: str
    value: float | str | None
    impact: float = Field(description="Contribution SHAP en log-odds (> 0 : augmente le risque de churn)")


class Explanation(BaseModel):
    risk_factors: list[Factor]
    protective_factors: list[Factor]


class Prediction(BaseModel):
    churn_probability: float
    action: Literal["RETENTION_OFFER", "NO_ACTION"]
    risk_band: Literal["Faible", "Modéré", "Élevé", "Très élevé"]
    threshold: float
    model_version: str
    explanation: Explanation | None = None


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[Customer] = Field(min_length=1)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_artifact():
    meta = json.loads(META_PATH.read_text())
    if not secrets.compare_digest(_sha256(MODEL_PATH), meta["model_sha256"]):
        raise RuntimeError("Empreinte du modèle invalide : artefact altéré, chargement refusé.")
    return joblib.load(MODEL_PATH), meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.meta = load_artifact()
    yield


docs_on = os.getenv("ENABLE_DOCS", "false").lower() == "true"
app = FastAPI(title="Telco Churn API", version="1.0.0", lifespan=lifespan, docs_url="/docs" if docs_on else None,
              redoc_url=None, openapi_url="/openapi.json" if docs_on else None)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API_KEY non configurée côté serveur.")
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Clé API invalide.")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    return response


def risk_band(p: float) -> str:
    return "Faible" if p < 0.15 else "Modéré" if p < 0.35 else "Élevé" if p < 0.6 else "Très élevé"


def _score(customers: list[Customer], explain: bool) -> list[Prediction]:
    meta = app.state.meta
    X = pd.DataFrame([c.model_dump() for c in customers])[meta["features"]]
    proba = app.state.model.predict_proba(X)[:, 1]
    exps = [None] * len(X)
    if explain:
        sv, _ = shap_values(app.state.model, X, meta["categorical_features"])
        exps = reason_codes(X, sv)
    thr = meta["threshold"]
    version = f"{meta['model']}-{meta['trained_at'][:10]}"
    return [Prediction(churn_probability=round(float(p), 4), action="RETENTION_OFFER" if p >= thr else "NO_ACTION",
                       risk_band=risk_band(float(p)), threshold=thr, model_version=version, explanation=e)
            for p, e in zip(proba, exps)]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.get("/model-info", dependencies=[Depends(require_api_key)])
def model_info():
    m = app.state.meta
    return {k: m[k] for k in ("model", "threshold", "campaign", "trained_at", "test_metrics", "n_train")}


@app.post("/predict", response_model=Prediction, dependencies=[Depends(require_api_key)])
def predict(customer: Customer, explain: bool = Query(True, description="Inclure les motifs (valeurs SHAP)")):
    return _score([customer], explain)[0]


@app.post("/predict/batch", response_model=list[Prediction], dependencies=[Depends(require_api_key)])
def predict_batch(batch: Batch, explain: bool = Query(False)):
    if len(batch.customers) > MAX_BATCH:
        raise HTTPException(413, f"Lot limité à {MAX_BATCH} clients.")
    return _score(batch.customers, explain)
