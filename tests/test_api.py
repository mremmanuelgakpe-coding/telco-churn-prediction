import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

os.environ["API_KEY"] = "test-key-not-for-prod"

from api.main import app  # noqa: E402

LOYAL = {"tenure": 60, "MonthlyCharges": 25.0, "TotalCharges": 1500.0, "SeniorCitizen": 0, "gender": "Male",
         "Partner": "Yes", "Dependents": "Yes", "PhoneService": "Yes", "MultipleLines": "No",
         "InternetService": "No", "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
         "DeviceProtection": "No internet service", "TechSupport": "No internet service",
         "StreamingTV": "No internet service", "StreamingMovies": "No internet service", "Contract": "Two year",
         "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)"}
RISKY = {"tenure": 3, "MonthlyCharges": 95.0, "TotalCharges": 285.0, "SeniorCitizen": 1, "gender": "Female",
         "Partner": "No", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "Yes",
         "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
         "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month",
         "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"}
H = {"X-API-Key": "test-key-not-for-prod"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_public(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_key_required(client):
    assert client.post("/predict", json=RISKY).status_code == 401
    assert client.post("/predict", json=RISKY, headers={"X-API-Key": "wrong"}).status_code == 401


def test_risky_scores_higher_and_is_targeted(client):
    loyal = client.post("/predict", json=LOYAL, headers=H).json()
    risky = client.post("/predict", json=RISKY, headers=H).json()
    assert risky["churn_probability"] > loyal["churn_probability"]
    assert risky["action"] == "RETENTION_OFFER" and loyal["action"] == "NO_ACTION"


@pytest.mark.parametrize("patch", [
    {"tenure": -1}, {"Contract": "Three year"}, {"unknown": 1},
    {"PhoneService": "No"},  # incohérent avec MultipleLines = Yes
    {"InternetService": "No"},  # incohérent avec les options Internet
])
def test_invalid_inputs(client, patch):
    assert client.post("/predict", json={**RISKY, **patch}, headers=H).status_code == 422


def test_explanation(client):
    exp = client.post("/predict", json=RISKY, headers=H).json()["explanation"]
    assert exp["risk_factors"] and all(f["impact"] > 0 for f in exp["risk_factors"])
    assert all(f["impact"] < 0 for f in exp["protective_factors"])
    assert client.post("/predict?explain=false", json=RISKY, headers=H).json()["explanation"] is None


def test_shap_sums_to_model_logit():
    from telco_churn.explain import shap_values

    model, meta = app.state.model, app.state.meta
    X = pd.DataFrame([LOYAL, RISKY])[meta["features"]]
    sv, base = shap_values(model, X, meta["categorical_features"])
    p = 1 / (1 + np.exp(-(base + sv.sum(axis=1).to_numpy())))
    assert np.allclose(p, model.predict_proba(X)[:, 1], atol=1e-6)


def test_batch_and_limit(client):
    r = client.post("/predict/batch", json={"customers": [LOYAL, RISKY]}, headers=H)
    assert r.status_code == 200 and len(r.json()) == 2
    assert client.post("/predict/batch", json={"customers": [LOYAL] * 501}, headers=H).status_code == 413


def test_docs_disabled_by_default(client):
    assert client.get("/docs").status_code == 404
