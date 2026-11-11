"""Configuration centrale : chemins, variables, hypothèses économiques de la campagne de rétention."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Empreinte du fichier Kaggle « blastchar/telco-customer-churn » (7 043 clients)
DATA_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"

ID_COL = "customerID"
TARGET = "Churn"  # 1 = le client a résilié dans le mois
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY = ["SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + BINARY + CATEGORICAL

# Hypothèses de la campagne de rétention (paramétrables, ce ne sont pas des données observées).
# Un client ciblé reçoit une offre qui coûte OFFER_MONTHS mois de son abonnement.
# Si c'est un vrai « churner », l'offre le retient avec une probabilité SUCCESS_RATE,
# et l'opérateur conserve RETAINED_MONTHS mois de revenus.
OFFER_MONTHS = 1.0
SUCCESS_RATE = 0.30
RETAINED_MONTHS = 12.0
