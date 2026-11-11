import numpy as np
import pandas as pd

from telco_churn.data import clean
from telco_churn.metrics import best_threshold, campaign_value


def test_clean_total_charges_blank():
    df = pd.DataFrame({"customerID": ["a", "b"], "TotalCharges": [" ", "10.5"], "tenure": [0, 3],
                       "Churn": ["No", "Yes"]})
    out, rep = clean(df)
    assert out["TotalCharges"].tolist() == [0.0, 10.5]
    assert rep["total_charges_blank_set_to_0"] == 1 and rep["blank_rows_all_tenure_0"]
    assert out["Churn"].tolist() == [0, 1]


def test_campaign_value_by_hand():
    # un churner ciblé (facture 100) : gain 0,3 × 12 × 100 = 360, coût 100 ; un non-churner ciblé : coût 50
    value, n = campaign_value([1, 0, 0], np.array([0.9, 0.8, 0.1]), [100, 50, 70], threshold=0.5)
    assert n == 2 and abs(value - (360 - 100 - 50)) < 1e-9


def test_threshold_targets_nobody_when_offer_too_expensive():
    y = [1, 0, 0, 0]
    p = np.array([0.9, 0.8, 0.7, 0.6])
    t, _ = best_threshold(y, p, [10, 10, 10, 10])
    assert t <= 0.9  # le churner à 0,9 rapporte plus qu'il ne coûte


def test_activity_cube_is_k_anonymous_and_labelled():
    import json

    import pandas as pd

    from telco_churn import config
    out = config.REPORTS_DIR / "activity"
    cube = pd.read_csv(out / "activity_cube.csv")
    manifest = json.loads((out / "manifest.json").read_text())
    assert cube["customers"].min() >= 5
    assert int(cube["customers"].sum()) == manifest["customers"] == 7043
    assert set(cube["risk_band"]) <= {"Faible", "Modéré", "Élevé", "Très élevé", "Regroupé"}
    assert set(cube["tenure_band"]) <= {"0-6 mois", "7-12 mois", "13-24 mois", "25-48 mois", "49-72 mois", "Regroupé"}
