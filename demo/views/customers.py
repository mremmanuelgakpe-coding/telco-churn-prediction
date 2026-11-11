"""Activité · Clients et revenu."""
import sys
from pathlib import Path

import altair as alt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (_on, click_bars, clickable, view, CONTRACT, DATA_BLUE, DATA_BLUE_LIGHT, GROUPED, INTERNET, ORANGE, PAYMENT, TENURE, card,  # noqa: E402
                    donut, filtered, footer, fr, heatmap, kpi_grid, money, pct, rate, scope_note, section)

d, excluded, sel = filtered()
scope_note(d, excluded, sel)
S = d.sum(numeric_only=True)
n, churned, mrr, lost = S["customers"], S["churned"], S["mrr"], S["mrr_churned"]
fine = d[d["internet"] != GROUPED]
F = fine.sum(numeric_only=True)
auto = fine[fine["payment"].isin(["Bank transfer (automatic)", "Credit card (automatic)"])]["customers"].sum()

section("Revenu")
kpi_grid([
    ("Revenu mensuel", money(mrr), f"{fr(n, 0)} clients facturés", "hl"),
    ("Facture moyenne", f"{fr(mrr / n, 2)} $", "par client et par mois"),
    ("Total facturé", money(S["total_charges"]), "cumul depuis l'entrée de chaque client"),
    ("Revenu mensuel perdu", money(lost), f"{pct(lost / mrr)} du revenu mensuel", "orange"),
])
section("Profil des abonnés")
kpi_grid([
    ("En couple", pct(S["partner"] / n, 0), f"{fr(S['partner'], 0)} clients"),
    ("Avec personnes à charge", pct(S["dependents"] / n, 0), f"{fr(S['dependents'], 0)} clients"),
    ("65 ans et plus", pct(fine.loc[fine["senior"] == "Senior", "customers"].sum() / max(F["customers"], 1), 0),
     "part des clients seniors"),
    ("Paiement automatique", pct(auto / max(F["customers"], 1), 0), "prélèvement ou carte bancaire"),
])

section("Paiement et ancienneté")
l, r = st.columns(2, gap="medium")
dp = view("payment")
pay = rate(dp, "payment", order=list(PAYMENT))
k = card(l, "Mode de paiement", "Répartition des clients et taux de churn de chaque mode. Cliquez pour filtrer.")
donut(k, [PAYMENT[c] for c in pay.index], list(pay["customers"]), [ORANGE, "#F8A26B", DATA_BLUE, DATA_BLUE_LIGHT][:len(pay)],
      fr(pay["customers"].sum(), 0), "clients", total=int(dp["customers"].sum()), dim="payment", codes=list(pay.index))
k.markdown("<div class='c-title' style='margin-top:.9rem'>Taux de churn par mode de paiement</div>", unsafe_allow_html=True)
click_bars(k, "payment", pay, key="cu_bars_payment")

ten_t = rate(view("tenure_band"), "tenure_band", order=TENURE)
ten = ten_t.reset_index().rename(columns={"index": "tenure_band"})
ten["cust_fr"], ten["rate_fr"] = ten["customers"].map(lambda v: fr(v, 0)), ten["rate"].map(lambda v: pct(v, 0))
ten["on"] = _on(ten, ["tenure_band"])
k = card(r, "Clients par ancienneté", "Nombre de clients dans chaque tranche d'ancienneté. Cliquez pour filtrer.")
base = alt.Chart(ten).encode(x=alt.X("tenure_band:N", sort=TENURE, title=None, axis=alt.Axis(labelAngle=0, labelOverlap=False)))
bars = base.mark_bar(color=DATA_BLUE, cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=46).encode(
    opacity=alt.Opacity("on:Q", scale=alt.Scale(domain=[0, 1], range=[0.25, 1]), legend=None),
    y=alt.Y("customers:Q", title="Clients", axis=alt.Axis(labelExpr="replace(format(datum.value, ',.0f'), /,/g, ' ')")),
    tooltip=[alt.Tooltip("tenure_band:N", title="Ancienneté"), alt.Tooltip("cust_fr:N", title="Clients"),
             alt.Tooltip("rate_fr:N", title="Churn")])
lab = base.mark_text(dy=-12, fontSize=14, fontWeight=600, color=DATA_BLUE).encode(y="customers:Q", text="cust_fr:N")
bars = bars.add_params(alt.selection_point(name="pick", fields=["tenure_band"]))
clickable(k, alt.layer(bars, lab), ["tenure_band"], "cu_tenure_volume", 250)
k.markdown("<div class='c-title' style='margin-top:.9rem'>Taux de churn par tranche</div>", unsafe_allow_html=True)
click_bars(k, "tenure_band", ten_t, key="cu_bars_tenure")

section("Revenu conservé et perdu")
l, r = st.columns(2, gap="medium")
m = d[d["contract"].isin(CONTRACT)].groupby("contract")[["mrr", "mrr_churned"]].sum().reindex(list(CONTRACT)).dropna()
k = card(l, "Par type de contrat", "Bleu : revenu mensuel des clients actifs. Orange : revenu des clients partis. "
                                    "À droite : part perdue.")
top = float(m["mrr"].max())
html = (f"<div class='slegend'><span><i style='background:{DATA_BLUE}'></i>Conservé</span>"
        f"<span><i style='background:{ORANGE}'></i>Perdu</span></div>")
for key, row in m.iterrows():
    keep, gone = row["mrr"] - row["mrr_churned"], row["mrr_churned"]
    wk, wl = keep / top * 100, gone / top * 100
    html += (f"<div class='srow'><div class='slab'>{CONTRACT[key]}<span>{money(keep)} conservés · {money(gone)} perdus</span>"
             f"</div><div class='strack'><div class='sbar' style='left:0;width:{wk:.1f}%;background:{DATA_BLUE};"
             f"border-radius:4px 0 0 4px'></div><div class='sbar' style='left:{wk:.1f}%;width:{wl:.1f}%;background:{ORANGE};"
             f"border-radius:0 4px 4px 0;border-left:2px solid #FFFFFF'></div></div>"
             f"<div class='sval num'>{pct(gone / row['mrr'], 0)}</div></div>")
k.markdown(html, unsafe_allow_html=True)

hv = view("contract", "internet")
g = hv[hv["internet"] != GROUPED].groupby(["contract", "internet"])[["mrr_churned"]].sum().reset_index()
g["Contrat"], g["Internet"] = g["contract"].map(CONTRACT), g["internet"].map(INTERNET)
k = card(r, "Revenu mensuel perdu : contrat × accès Internet", "Plus la case est foncée, plus la perte est forte. "
                                                               "Cliquez sur une case pour filtrer.")
heatmap(k, g, "Internet", "Contrat", "mrr_churned", list(INTERNET.values()), list(CONTRACT.values()),
        g["mrr_churned"].map(money), dims=("contract", "internet"), key="cu_heat")
footer()
