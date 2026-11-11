"""Activité · Churn par segment : grilles croisées."""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (click_bars, view, CONTRACT, CONTRACT_SHORT, GROUPED, INTERNET, PAYMENT, RISK, SENIOR, TENURE, bar_rows, card, filtered, footer, fr,  # noqa: E402
                    heatmap, kpi_grid, money, pct, rate, scope_note, section)

d, excluded, sel = filtered()
scope_note(d, excluded, sel)
S = d.sum(numeric_only=True)
n, churned = S["customers"], S["churned"]
fine = d[d["internet"] != GROUPED]

seg = fine.groupby(["contract", "internet"])[["customers", "churned"]].sum()
seg = seg[seg["customers"] >= 30]
seg["rate"] = seg["churned"] / seg["customers"]


def seg_name(idx):
    return f"{CONTRACT[idx[0]]} · {INTERNET[idx[1]]}"


section("Vue d'ensemble")
items = [("Taux de churn", pct(churned / n), f"{fr(churned, 0)} départs sur {fr(n, 0)} clients", "hl"),
         ("Revenu mensuel perdu", money(S["mrr_churned"]), f"{pct(S['mrr_churned'] / S['mrr'])} du revenu", "orange")]
if len(seg):
    hi, lo = seg["rate"].idxmax(), seg["rate"].idxmin()
    items += [("Segment le plus exposé", pct(seg.loc[hi, "rate"], 0), f"{seg_name(hi)} ({fr(seg.loc[hi, 'customers'], 0)} clients)",
               "orange"),
              ("Segment le plus fidèle", pct(seg.loc[lo, "rate"], 0) if len(seg) > 1 else "—",
               f"{seg_name(lo)} ({fr(seg.loc[lo, 'customers'], 0)} clients)" if len(seg) > 1
               else "un seul segment dans le périmètre", "blue")]
kpi_grid(items)


def grid(dim_x, dim_y, x_names, y_names, x_order, y_order):
    v = view(dim_x, dim_y)
    g = v[v["internet"] != GROUPED].groupby([dim_y, dim_x])[["customers", "churned"]].sum().reset_index()
    g = g[g["customers"] >= 5]
    g["rate"] = g["churned"] / g["customers"]
    g["X"] = g[dim_x].map(lambda v: x_names.get(v, v))
    g["Y"] = g[dim_y].map(lambda v: y_names.get(v, v))
    return g, [x_names.get(v, v) for v in x_order], [y_names.get(v, v) for v in y_order]


section("Grilles de churn", "Chaque case donne le taux de départ du segment croisé (cases de moins de 5 clients masquées). "
        "Cliquez sur une case pour filtrer tout le tableau de bord.")
k = card(None, "Contrat × ancienneté", "Taux de départ selon l'engagement et le temps passé chez l'opérateur.")
g, xo, yo = grid("tenure_band", "contract", {}, CONTRACT, TENURE, list(CONTRACT))
heatmap(k, g, "X", "Y", "rate", xo, yo, g["rate"].map(lambda v: pct(v, 0)), scale_max=0.6,
        dims=("tenure_band", "contract"), key="sg_heat_tenure")

l, r = st.columns(2, gap="medium")
k = card(l, "Mode de paiement × contrat", "Taux de départ selon le moyen de paiement et l'engagement.")
g, xo, yo = grid("contract", "payment", CONTRACT_SHORT, PAYMENT, list(CONTRACT), list(PAYMENT))
heatmap(k, g, "X", "Y", "rate", xo, yo, g["rate"].map(lambda v: pct(v, 0)), scale_max=0.6, dims=("contract", "payment"),
        key="sg_heat_payment")
k = card(r, "Âge × accès Internet", "Taux de départ selon l'âge et le type d'accès.")
g, xo, yo = grid("internet", "senior", INTERNET, SENIOR, list(INTERNET), list(SENIOR))
heatmap(k, g, "X", "Y", "rate", xo, yo, g["rate"].map(lambda v: pct(v, 0)), scale_max=0.6, dims=("internet", "senior"),
        key="sg_heat_age")

section("Classement des segments")
l, r = st.columns(2, gap="medium")
k = card(l, "Taux de churn par critère")
dims = {"Contrat": ("contract", list(CONTRACT), CONTRACT), "Accès Internet": ("internet", list(INTERNET), INTERNET),
        "Paiement": ("payment", list(PAYMENT), PAYMENT), "Ancienneté": ("tenure_band", TENURE, None),
        "Âge": ("senior", list(SENIOR), SENIOR), "Niveau de risque": ("risk_band", RISK, None)}
choice = k.segmented_control("Critère", list(dims), default="Paiement", label_visibility="collapsed", key="seg_dim")
col, order, names = dims[choice or "Paiement"]
t = rate(view(col), col, order=order)
click_bars(k, col, t, sub=lambda r: f"{fr(r['customers'], 0)} clients", key=f"sg_bars_{col}")
k = card(r, "Les 8 segments les plus exposés", "Contrat × accès Internet × paiement, segments d'au moins 30 clients.")
top = fine.groupby(["contract", "internet", "payment"])[["customers", "churned"]].sum()
top = top[top["customers"] >= 30]
top["rate"] = top["churned"] / top["customers"]
top = top.sort_values("rate", ascending=False).head(8)
bar_rows(k, [(f"{INTERNET[i]} · {CONTRACT_SHORT[c].lower()}", f"{PAYMENT[p]} · {fr(row['customers'], 0)} clients",
              row["rate"]) for (c, i, p), row in top.iterrows()], max_value=1, wide=True)
footer()
