"""Test de fumée de la démo : elle s'exécute sans erreur sur les trois profils et sans Internet."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "demo" / "app.py")


def run(at):
    at.run(timeout=90)
    assert not at.exception, at.exception
    return at


def test_presets():
    at = run(AppTest.from_file(APP, default_timeout=90))
    for i in range(3):
        at.button[i].click()
        run(at)
    assert at.session_state["contract"] == "Month-to-month"


def test_no_internet_profile():
    at = run(AppTest.from_file(APP, default_timeout=90))
    at.button[0].click()  # client fidèle : pas d'Internet
    run(at)
    assert at.session_state["internet"] == "No"
