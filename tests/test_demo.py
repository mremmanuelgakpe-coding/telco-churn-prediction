"""Tests de fumée du tableau de bord : chaque page s'exécute sans erreur, avec et sans filtres, sans Internet."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parents[1] / "demo" / "app.py")
PAGES = ["views/overview.py", "views/customers.py", "views/services.py", "views/segments.py", "views/risk.py"]


def run(at):
    at.run(timeout=90)
    assert not at.exception, at.exception
    return at


def page(path):
    at = run(AppTest.from_file(APP, default_timeout=90))
    at.switch_page(path)
    return run(at)


@pytest.mark.parametrize("path", PAGES)
def test_activity_pages(path):
    at = page(path)
    assert any("kgrid" in m.value for m in at.markdown)


@pytest.mark.parametrize("path", PAGES)
def test_activity_pages_with_filters(path):
    at = page(path)
    at.multiselect(key="flt_contract").set_value(["Month-to-month"])
    at.multiselect(key="flt_internet").set_value(["Fiber optic"])
    run(at)
    assert any("regroupés pour" in m.value for m in at.markdown)


def test_filter_without_internet_clients():
    at = page("views/services.py")
    at.multiselect(key="flt_internet").set_value(["No"])
    run(at)


def test_presets():
    at = page("views/model.py")
    for i in range(3):
        at.button[i].click()
        run(at)
    assert at.session_state["contract"] == "Month-to-month"


def test_no_internet_profile():
    at = page("views/model.py")
    at.button[0].click()  # client fidèle : pas d'Internet
    run(at)
    assert at.session_state["internet"] == "No"
