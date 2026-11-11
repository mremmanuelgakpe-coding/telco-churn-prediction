"""Rétention · Clients à risque (score du modèle appliqué à la base)."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (click_bars, view, BLUE_RAMP, CONTRACT, GROUPED, INTERNET, RISK, RISK_COLORS, card, donut, filtered,  # noqa: E402
                    footer, fr, heatmap, kpi_grid, load_activity, money, pct, rate, scope_note, section)

d, excluded, sel = filtered()
scope_note(d, excluded, sel)
_, _, manifest = load_activity()
THR = manifest["threshold"]
S = d.sum(numeric_only=True)
n, churned, mrr, lost = S["customers"], S["churned"], S["mrr"], S["mrr_churned"]
active, mrr_active = n - churned, mrr - lost
tgt, mrr_tgt = S["to_target_active"], S["mrr_to_target_active"]

section("Cible de la campagne", f"Clients encore actifs dont le score dépasse le seuil de {pct(THR, 0)} retenu par le modèle.")
kpi_grid([
    ("Clients actifs à cibler", fr(tgt, 0), f"{pct(tgt / max(active, 1), 0)} des {fr(active, 0)} clients actifs", "hl"),
    ("Revenu mensuel à risque", money(mrr_tgt), f"{pct(mrr_tgt / max(mrr_active, 1), 0)} du revenu des actifs", "orange"),
    ("Revenu annuel à sécuriser", money(12 * mrr_tgt), "revenu mensuel à risque × 12", "orange"),
    ("Probabilité moyenne de départ", pct(S["score_sum"] / n), f"à comparer au churn observé de {pct(churned / n)}",
     "blue"),
])

rb = rate(d, "risk_band", order=RISK)
rb["active"] = rb["customers"] - rb["churned"]
rb["mrr_active"] = rb["mrr"] - rb["mrr_churned"]

section("Portefeuille par niveau de risque")
cards = []
for band, row in rb.iterrows():
    tone = {"Faible": "", "Modéré": "", "Élevé": "orange", "Très élevé": "hl"}[band]
    cards.append((band, f"{fr(row['active'], 0)} actifs", f"{money(row['mrr_active'])} de revenu mensuel · churn observé "
                                                          f"{pct(row['rate'], 0)}", tone))
kpi_grid(cards)

l, r = st.columns(2, gap="medium")
dr = view("risk_band")
rv = rate(dr, "risk_band", order=RISK)
rv["active"] = rv["customers"] - rv["churned"]
k = card(l, "Clients actifs par niveau de risque", "Répartition des clients encore abonnés. Cliquez pour filtrer.")
donut(k, list(rv.index), list(rv["active"]), [RISK_COLORS[b] for b in rv.index], fr(rv["active"].sum(), 0), "clients actifs",
      total=int((dr["customers"] - dr["churned"]).sum()), dim="risk_band", codes=list(rv.index), key="rk_donut")
k = card(r, "Le score classe-t-il bien les clients ?", "Taux de départ observé dans chaque niveau de risque. "
                                                      "Cliquez pour filtrer.")
click_bars(k, "risk_band", rv, sub=lambda r: f"{fr(r['customers'], 0)} clients", key="rk_bars")
k.markdown("<div class='note'>Le taux de départ monte nettement d'un niveau à l'autre : les clients classés « Très élevé » "
           "partent bien plus que les « Faible ». Ces scores couvrent toute la base, dont 80 % de clients vus à "
           "l'entraînement : c'est un usage de pilotage. La performance du modèle se lit sur le jeu de test, dans la page "
           "« Modèle et campagne ».</div>", unsafe_allow_html=True)

section("Où sont les clients à cibler")
hv = view("contract", "internet")
fine = hv[hv["internet"] != GROUPED]
g = fine.groupby(["contract", "internet"])[["to_target_active", "mrr_to_target_active"]].sum().reset_index()
g["Contrat"], g["Internet"] = g["contract"].map(CONTRACT), g["internet"].map(INTERNET)
l, r = st.columns(2, gap="medium")
k = card(l, "Clients actifs à cibler", "Contrat × accès Internet. Cliquez sur une case pour filtrer.")
heatmap(k, g, "Internet", "Contrat", "to_target_active", list(INTERNET.values()), list(CONTRACT.values()),
        g["to_target_active"].map(lambda v: fr(v, 0)), dims=("contract", "internet"), key="rk_heat_n")
k = card(r, "Revenu mensuel à risque", "Contrat × accès Internet. Cliquez sur une case pour filtrer.")
heatmap(k, g, "Internet", "Contrat", "mrr_to_target_active", list(INTERNET.values()), list(CONTRACT.values()),
        g["mrr_to_target_active"].map(money), ramp=BLUE_RAMP, dims=("contract", "internet"), key="rk_heat_mrr")

with st.container(key="tomodel"):
    st.page_link("views/model.py", label="Simuler un client et régler la campagne de rétention",
                 icon=":material/arrow_forward:")
footer()
