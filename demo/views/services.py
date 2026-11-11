"""Activité · Services souscrits et churn."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (click_bars, view, BLUE_RAMP, CONTRACT, DATA_BLUE, DATA_BLUE_LIGHT, INTERNET, NEUTRAL, ORANGE, card, donut,  # noqa: E402
                    filtered, footer, fr, heatmap, kpi_grid, pct, rate, scope_note, section)

OPTIONS = {"OnlineSecurity": "Sécurité en ligne", "TechSupport": "Support technique", "OnlineBackup": "Sauvegarde en ligne",
           "DeviceProtection": "Protection de l'appareil", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming films"}

d, excluded, sel = filtered()
scope_note(d, excluded, sel)
S = d.sum(numeric_only=True)
n = S["customers"]
net_n = S["fiber"] + S["dsl"]
phone, multi = S["phone"], S["multiple_lines"]

section("Souscriptions")
kpi_grid([
    ("Ligne téléphonique", pct(phone / n, 0), f"{fr(phone, 0)} clients"),
    ("Lignes multiples", pct(multi / max(phone, 1), 0), "parmi les clients avec téléphone"),
    ("Clients Internet", pct(net_n / n, 0), f"fibre {pct(S['fiber'] / n, 0)} · ADSL {pct(S['dsl'] / n, 0)}", "hl"),
    ("Options par client Internet", fr(S["options_sum"] / max(net_n, 1), 1), "sur 6 options disponibles"),
])

section("Équipement")
l, r = st.columns(2, gap="medium")
di = view("internet")
V = di.sum(numeric_only=True)
k = card(l, "Accès Internet", "Répartition des clients et taux de churn de chaque accès. Cliquez pour filtrer.")
donut(k, ["Fibre", "ADSL", "Sans Internet"], [V["fiber"], V["dsl"], V["customers"] - V["fiber"] - V["dsl"]],
      [ORANGE, DATA_BLUE, NEUTRAL], fr(V["customers"], 0), "clients", dim="internet", codes=list(INTERNET), key="sv_donut")
k.markdown("<div class='c-title' style='margin-top:.9rem'>Taux de churn par accès</div>", unsafe_allow_html=True)
click_bars(k, "internet", rate(di, "internet", order=list(INTERNET)), key="sv_bars")
k = card(r, "Téléphonie", "Clients sans ligne, avec une ligne ou plusieurs lignes.")
donut(k, ["Sans ligne", "Une ligne", "Lignes multiples"], [n - phone, phone - multi, multi], [NEUTRAL, DATA_BLUE_LIGHT, DATA_BLUE],
      fr(n, 0), "clients")

section("Options Internet", "Taux d'adoption parmi les clients Internet, et taux de churn des clients avec et sans l'option.")
net = d[d["internet"].isin(["Fiber optic", "DSL"])]
N = net.sum(numeric_only=True)
if N.get("customers", 0) == 0:
    st.info("Aucun client Internet dans le périmètre filtré.")
else:
    cards = ""
    for key, name in OPTIONS.items():
        has, churn_has = N[f"opt_{key}"], N[f"churn_opt_{key}"]
        without = N["customers"] - has
        r_has = churn_has / max(has, 1)
        r_without = (N["churned"] - churn_has) / max(without, 1)
        color = ORANGE if r_has > r_without else DATA_BLUE
        cards += (f"<div class='ocard'><div class='o-name'>{name}</div>"
                  f"<div class='o-val num'>{pct(has / N['customers'], 0)}<small>des clients Internet</small></div>"
                  f"<div class='otrack'><div style='width:{has / N['customers'] * 100:.1f}%'></div></div>"
                  f"<div class='o-cmp'><span>Churn avec : <b style='color:{color}'>{pct(r_has, 0)}</b></span>"
                  f"<span>sans : <b>{pct(r_without, 0)}</b></span></div></div>")
    st.markdown(f"<div class='ogrid'>{cards}</div>", unsafe_allow_html=True)
    st.markdown("<div class='note'>Les options de sécurité et de support vont avec beaucoup moins de départs ; le streaming "
                "n'a pas cet effet. Corrélation observée, pas un effet causal démontré.</div>", unsafe_allow_html=True)

section("Croisements")
l, r = st.columns(2, gap="medium")
hv = view("contract", "internet")
fine = hv[hv["internet"] != "Regroupé"]
g = fine.groupby(["contract", "internet"])[["customers", "churned"]].sum().reset_index()
g = g[g["customers"] >= 5]
g["rate"] = g["churned"] / g["customers"]
g["Contrat"], g["Internet"] = g["contract"].map(CONTRACT), g["internet"].map(INTERNET)
k = card(l, "Taux de churn : contrat × accès Internet", "Taux de départ de chaque segment. Cliquez sur une case pour filtrer.")
heatmap(k, g, "Internet", "Contrat", "rate", list(INTERNET.values()), list(CONTRACT.values()), g["rate"].map(lambda v: pct(v, 0)),
        scale_max=0.6, dims=("contract", "internet"), key="sv_heat_rate")
k = card(r, "Nombre de clients : contrat × accès Internet", "Taille de chaque segment. Cliquez sur une case pour filtrer.")
heatmap(k, g, "Internet", "Contrat", "customers", list(INTERNET.values()), list(CONTRACT.values()),
        g["customers"].map(lambda v: fr(v, 0)), ramp=BLUE_RAMP, dims=("contract", "internet"), key="sv_heat_count")
footer()
