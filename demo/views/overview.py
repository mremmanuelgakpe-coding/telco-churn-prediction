"""Synthèse : la page d'accueil de la direction et du marketing."""
import sys
from pathlib import Path

import altair as alt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (click_bars, view, CONTRACT, DATA_BLUE, DATA_BLUE_LIGHT, INTERNET, NEUTRAL, ORANGE, PAYMENT, card, chart_config,  # noqa: E402
                    donut, filtered, footer, fr, insights, kpi_grid, load_activity, money, pct, rate, scope_note,
                    section)

d, excluded, sel = filtered()
scope_note(d, excluded, sel)
_, curve, manifest = load_activity()
S = d.sum(numeric_only=True)
n, churned, mrr, lost = S["customers"], S["churned"], S["mrr"], S["mrr_churned"]
active = n - churned

section("Base clients")
kpi_grid([
    ("Clients", fr(n, 0), f"{fr(active, 0)} actifs · {fr(churned, 0)} partis"),
    ("Taux de churn", pct(churned / n), "part des clients partis dans le mois", "hl"),
    ("Revenu mensuel", money(mrr), f"facture moyenne {fr(mrr / n, 2)} $"),
    ("Ancienneté moyenne", f"{fr(S['tenure_sum'] / n, 1)} mois", f"total facturé {money(S['total_charges'])}"),
])
section("Rétention")
arpu_lost = lost / max(churned, 1)
arpu_active = (mrr - lost) / max(active, 1)
kpi_grid([
    ("Revenu mensuel perdu", money(lost), f"{pct(lost / mrr)} du revenu mensuel", "orange"),
    ("Facture moyenne des partants", f"{fr(arpu_lost, 2)} $", f"contre {fr(arpu_active, 2)} $ pour les actifs",
     "orange"),
    ("Clients actifs à cibler", fr(S["to_target_active"], 0),
     f"{pct(S['to_target_active'] / max(active, 1), 0)} des actifs (score ≥ {pct(manifest['threshold'], 0)})", "blue"),
    ("Revenu mensuel à risque", money(S["mrr_to_target_active"]),
     f"{pct(S['mrr_to_target_active'] / max(mrr - lost, 1), 0)} du revenu des actifs", "orange"),
])

section("Répartition de la base")
c1, c2, c3 = st.columns(3, gap="medium")
k = card(c1, "Statut des clients", "Clients encore abonnés et clients partis.")
donut(k, ["Actifs", "Partis"], [active, churned], [DATA_BLUE, ORANGE], fr(n, 0), "clients")
ct_all = rate(view("contract"), "contract", order=list(CONTRACT))
k = card(c2, "Type de contrat", "Nombre de clients par engagement. Cliquez pour filtrer.")
donut(k, [CONTRACT[c] for c in ct_all.index], list(ct_all["customers"]), [ORANGE, DATA_BLUE, DATA_BLUE_LIGHT][:len(ct_all)],
      fr(ct_all["customers"].sum(), 0), "clients", dim="contract", codes=list(ct_all.index))
V = view("internet").sum(numeric_only=True)
k = card(c3, "Accès Internet", "Fibre, ADSL ou téléphonie seule. Cliquez pour filtrer.")
donut(k, ["Fibre", "ADSL", "Sans Internet"], [V["fiber"], V["dsl"], V["customers"] - V["fiber"] - V["dsl"]],
      [ORANGE, DATA_BLUE, NEUTRAL], fr(V["customers"], 0), "clients", dim="internet", codes=list(INTERNET))
ct = rate(d, "contract", order=list(CONTRACT), names=CONTRACT)

section("Où partent les clients")
l, r = st.columns(2, gap="medium")
k = card(l, "Taux de churn par contrat", "Part des clients partis. Cliquez sur une barre pour filtrer.")
click_bars(k, "contract", ct_all, sub=lambda r: f"{fr(r['customers'], 0)} clients", key="ov_bars_contract")
k.markdown("<div class='c-title' style='margin-top:.6rem'>Taux de churn par accès Internet</div>", unsafe_allow_html=True)
click_bars(k, "internet", rate(view("internet"), "internet", order=list(INTERNET)),
           sub=lambda r: f"{fr(r['customers'], 0)} clients", key="ov_bars_internet")

k = card(r, "Churn selon l'ancienneté", "Taux de départ par mois d'ancienneté (filtres contrat et Internet appliqués).")
c = curve.copy()
if sel["contract"]:
    c = c[c["contract"].isin(sel["contract"])]
if sel["internet"]:
    c = c[c["internet"].isin(sel["internet"])]
c = c.groupby("tenure")[["customers", "churned"]].sum().reset_index()
c = c[c["customers"] >= 5]
c["rate"] = c["churned"] / c["customers"]
area = alt.Chart(c).mark_area(color=ORANGE, opacity=0.12, interpolate="linear").encode(
    x=alt.X("tenure:Q", title="Ancienneté (mois)", scale=alt.Scale(domain=[0, 72], nice=False)), y="rate:Q")
line = alt.Chart(c).mark_line(color=ORANGE, strokeWidth=2.5).encode(
    x=alt.X("tenure:Q", title="Ancienneté (mois)", scale=alt.Scale(domain=[0, 72], nice=False)),
    y=alt.Y("rate:Q", title="Taux de churn", axis=alt.Axis(tickCount=5, labelExpr="format(datum.value * 100, '.0f') + ' %'")),
    tooltip=[alt.Tooltip("tenure:Q", title="Ancienneté (mois)"), alt.Tooltip("rate:Q", title="Churn", format=".1%"),
             alt.Tooltip("customers:Q", title="Clients")])
k.altair_chart(chart_config(area + line, 330), use_container_width=True)

section("Points d'attention")
k = card()
items = []
if len(ct) >= 2:
    items.append(f"Les clients <b>{ct['rate'].idxmax().lower()}</b> partent à <b>{pct(ct['rate'].max(), 0)}</b>, contre "
                 f"<b>{pct(ct['rate'].min(), 0)}</b> pour les clients <b>{ct['rate'].idxmin().lower()}</b>.")
first = c[c["tenure"].between(1, 6)]
if not first.empty and first["customers"].sum() and active:
    r6 = first["churned"].sum() / first["customers"].sum()
    items.append(f"Les six premiers mois sont les plus risqués : <b>{pct(r6, 0)}</b> de départs, contre "
                 f"<b>{pct(churned / n, 0)}</b> en moyenne.")
if churned:
    items.append(f"Les clients partis payaient en moyenne <b>{fr(arpu_lost, 2)} $</b> par mois, "
                 f"<b>{pct(arpu_lost / arpu_active - 1, 0)}</b> de plus que les clients restés : le churn touche les "
                 f"meilleurs revenus.")
pay = rate(d, "payment")
if len(pay) >= 2:
    items.append(f"Mode de paiement le plus exposé : <b>{PAYMENT.get(pay['rate'].idxmax())}</b> "
                 f"(<b>{pct(pay['rate'].max(), 0)}</b> de churn).")
items.append(f"<b>{fr(S['to_target_active'], 0)}</b> clients actifs dépassent le seuil du modèle : ils représentent "
             f"<b>{money(S['mrr_to_target_active'])}</b> de revenu mensuel à sécuriser (voir « Clients à risque »).")
insights(k, items)
with st.container(key="tomodel"):
    st.page_link("views/risk.py", label="Voir les clients à risque et la campagne de rétention", icon=":material/arrow_forward:")
footer()
