"""Vue « Modèle » : simulateur client, campagne de rétention, performance et méthode."""
from __future__ import annotations

import html
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import (DATA_BLUE, GITHUB_URL, INK, MUTED, ORANGE, ORANGE_RAMP, chart_config, footer, fr,  # noqa: E402
                    load_model, money, pct)
from telco_churn.explain import LABELS_FR, shap_values  # noqa: E402

BLUE = DATA_BLUE
GREY = "#D9CFC4"
BANDS = [("Faible", 0.00, 0.15), ("Modéré", 0.15, 0.35), ("Élevé", 0.35, 0.60), ("Très élevé", 0.60, 1.00)]
BAND_COLORS = ["#FDE3D0", "#FBC9A6", "#F8A26B", "#F4721E"]
MODEL_LABEL = {"logistic_regression": "Régression logistique", "svm": "SVM (RBF)", "knn": "KNN",
               "naive_bayes": "Naive Bayes", "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting",
               "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost"}


# ---------- Profils pré-remplis ----------
SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
PRESETS = {
    "Client fidèle": dict(tenure=58, monthly=24.5, contract="Two year", internet="No", payment="Bank transfer (automatic)",
                          phone=True, multi=False, paperless=False, senior=False, partner=True, dependents=True,
                          gender="Male", **{s: False for s in SERVICES}),
    "Client hésitant": dict(tenure=14, monthly=70.0, contract="One year", internet="DSL", payment="Mailed check",
                            phone=True, multi=False, paperless=True, senior=False, partner=True, dependents=False,
                            gender="Female", OnlineSecurity=True, OnlineBackup=False, DeviceProtection=False,
                            TechSupport=True, StreamingTV=True, StreamingMovies=False),
    "Client à risque": dict(tenure=4, monthly=94.5, contract="Month-to-month", internet="Fiber optic",
                            payment="Electronic check", phone=True, multi=True, paperless=True, senior=True,
                            partner=False, dependents=False, gender="Female", OnlineSecurity=False, OnlineBackup=False,
                            DeviceProtection=False, TechSupport=False, StreamingTV=True, StreamingMovies=True),
}
for k, v in PRESETS["Client hésitant"].items():
    st.session_state.setdefault(k, v)


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


CONTRACT = {"Month-to-month": "Sans engagement (mensuel)", "One year": "Engagement 1 an", "Two year": "Engagement 2 ans"}
INTERNET = {"Fiber optic": "Fibre", "DSL": "ADSL", "No": "Pas d'Internet"}
PAYMENT = {"Electronic check": "Chèque électronique", "Mailed check": "Chèque papier",
           "Bank transfer (automatic)": "Prélèvement bancaire", "Credit card (automatic)": "Carte bancaire (auto)"}
SERVICE_FR = {"OnlineSecurity": "Sécurité en ligne", "OnlineBackup": "Sauvegarde en ligne",
              "DeviceProtection": "Protection de l'appareil", "TechSupport": "Support technique",
              "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming films"}
VALUE_FR = {**CONTRACT, **INTERNET, **PAYMENT, "Yes": "Oui", "No": "Non", "No internet service": "Sans Internet",
            "No phone service": "Sans téléphone", "Female": "Femme", "Male": "Homme"}


def fmt_value(feat, v):
    if feat == "MonthlyCharges" or feat == "TotalCharges":
        return money(v) if feat == "TotalCharges" else f"{fr(v, 2)} $ / mois"
    if feat == "tenure":
        return f"{v:.0f} mois"
    if feat == "SeniorCitizen":
        return "Oui" if v == 1 else "Non"
    return VALUE_FR.get(v, str(v))


model, meta, report, camp = load_model()
tab_sim, tab_camp, tab_perf, tab_method = st.tabs(["Simulateur", "Campagne de rétention", "Performance du modèle", "Méthode"])

# =====================================================================
with tab_sim:
    left, right = st.columns([0.95, 1.35], gap="medium")
    with left:
        with st.container(key="form"):
            st.markdown("<h3>Profil de l'abonné</h3>", unsafe_allow_html=True)
            b = st.columns(3)
            for col, name in zip(b, PRESETS):
                col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)
            c1, c2 = st.columns(2)
            c1.number_input("Ancienneté (mois)", 0, 72, key="tenure", step=1)
            c2.number_input("Facture mensuelle ($)", 18.0, 120.0, key="monthly", step=0.5, format="%.2f")
            st.selectbox("Contrat", list(CONTRACT), format_func=CONTRACT.get, key="contract")
            c3, c4 = st.columns(2)
            c3.selectbox("Accès Internet", list(INTERNET), format_func=INTERNET.get, key="internet")
            c4.selectbox("Paiement", list(PAYMENT), format_func=PAYMENT.get, key="payment")
            has_net = st.session_state.internet != "No"
            st.markdown("<div class='sub' style='margin:.3rem 0 .2rem'>Options Internet</div>", unsafe_allow_html=True)
            o1, o2 = st.columns(2)
            for i, s in enumerate(SERVICES):
                (o1 if i % 2 == 0 else o2).checkbox(SERVICE_FR[s], key=s, disabled=not has_net)
            st.markdown("<div class='sub' style='margin:.3rem 0 .2rem'>Téléphonie et profil</div>", unsafe_allow_html=True)
            p1, p2 = st.columns(2)
            p1.checkbox("Ligne téléphonique", key="phone")
            p2.checkbox("Lignes multiples", key="multi", disabled=not st.session_state.phone)
            p1.checkbox("Facture dématérialisée", key="paperless")
            p2.checkbox("Client senior", key="senior")
            p1.checkbox("En couple", key="partner")
            p2.checkbox("Personnes à charge", key="dependents")
            st.radio("Genre", ["Female", "Male"], format_func=VALUE_FR.get, key="gender", horizontal=True)

    s = st.session_state
    net = s.internet != "No"
    yn = lambda b: "Yes" if b else "No"  # noqa: E731
    row = {
        "tenure": int(s.tenure), "MonthlyCharges": float(s.monthly),
        "TotalCharges": float(round(s.tenure * s.monthly, 2)), "SeniorCitizen": int(s.senior), "gender": s.gender,
        "Partner": yn(s.partner), "Dependents": yn(s.dependents), "PhoneService": yn(s.phone),
        "MultipleLines": yn(s.multi) if s.phone else "No phone service", "InternetService": s.internet,
        **{k: (yn(s[k]) if net else "No internet service") for k in SERVICES},
        "Contract": s.contract, "PaperlessBilling": yn(s.paperless), "PaymentMethod": s.payment,
    }
    X = pd.DataFrame([row])[meta["features"]]
    p = float(model.predict_proba(X)[:, 1][0])
    sv, base = shap_values(model, X, meta["categorical_features"])
    contrib = sv.iloc[0]
    thr = meta["threshold"]
    target = p >= thr
    band_i = next(i for i, (_, lo, hi) in enumerate(BANDS) if lo <= p < hi or (i == 3 and p >= lo))
    c = meta["campaign"]
    expected = p * c["success_rate"] * c["retained_months"] * row["MonthlyCharges"] - c["offer_months"] * row["MonthlyCharges"]

    with right:
        k1, k2 = st.columns(2, gap="small")
        k1.markdown(f"""<div class="hero"><span class="badge">seuil {pct(thr, 0)}</span><span class="icon">%</span>
        <div class="label">Probabilité de résiliation</div><div class="value num">{pct(p)}</div></div>""",
                    unsafe_allow_html=True)
        cls, word = ("ko", "Proposer l'offre") if target else ("ok", "Pas d'action")
        k2.markdown(f"""<div class="card"><span class="badge light">{money(row['TotalCharges'])} facturés</span>
        <span class="icon">✓</span><div class="label">Action recommandée</div><div class="decision {cls}">{word}</div>
        <div class="sub">Valeur attendue de l'offre pour ce client : <b style="color:#1E2A3B">{money(expected)}</b>
        (hypothèses de l'onglet Campagne).</div></div>""", unsafe_allow_html=True)
        segs = "".join(
            f'<div class="seg{" on" if i == band_i else ""}" style="background:{BAND_COLORS[i]}" '
            f'title="{b} : {pct(lo, 0)} à {pct(hi, 0)}"><span>{b}</span></div>' for i, (b, lo, hi) in enumerate(BANDS))
        st.markdown(f"""<div class="card" style="margin-top:12px; padding-bottom:1.9rem">
        <div class="scale-top"><div><span style="font-size:1rem;color:#1E2A3B;font-weight:500">Niveau de risque</span>
        <div class="sub">de {pct(BANDS[band_i][1], 0)} à {pct(BANDS[band_i][2], 0)} de probabilité</div></div>
        <b>{BANDS[band_i][0]}</b></div><div class="scale">{segs}</div></div>""", unsafe_allow_html=True)

        ranked = contrib.sort_values()
        ups = [f for f in ranked.index[::-1] if ranked[f] > 0][:3]
        downs = [f for f in ranked.index if ranked[f] < 0][:3]

        def line(f, up):
            return (f'<div class="reason"><div><div class="t">{html.escape(LABELS_FR[f])}</div>'
                    f'<div class="v">{html.escape(fmt_value(f, row[f]))}</div></div>'
                    f'<div class="dot {"up" if up else "down"}">{"▲" if up else "▼"}</div></div>')

        body = ""
        if ups:
            body += '<div class="group">Ce qui pousse au départ</div>' + "".join(line(f, True) for f in ups)
        if downs:
            body += '<div class="group">Ce qui retient le client</div>' + "".join(line(f, False) for f in downs)
        st.markdown(f"""<div class="dark" style="margin-top:12px"><span class="count">{len(ups) + len(downs)} motifs</span>
        <h3>Pourquoi ce score</h3><div class="hint">Valeurs SHAP calculées pour ce client.</div>{body}</div>""",
                    unsafe_allow_html=True)

    with st.container(key="chart"):
        st.markdown("<h3>Contribution de chaque variable</h3>", unsafe_allow_html=True)
        st.caption(f"Le modèle part d'un score de base de {pct(1 / (1 + np.exp(-base[0])), 0)}. Chaque barre montre "
                   "de combien la variable pousse le risque vers le haut ou vers le bas (SHAP, en log-odds). "
                   f"Total facturé calculé : ancienneté × facture mensuelle = {money(row['TotalCharges'])}.")
        order = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
        vmax = max(float(order.abs().max()), 1e-9)
        rows_html = ""
        for f, v in order.items():
            if abs(v) < 0.005:
                continue
            w = abs(v) / vmax * 50
            side = f"left:50%;width:{w:.1f}%" if v > 0 else f"left:{50 - w:.1f}%;width:{w:.1f}%"
            color = ORANGE if v > 0 else BLUE
            rows_html += (f'<div class="srow"><div class="slab">{html.escape(LABELS_FR[f])}'
                          f'<span>{html.escape(fmt_value(f, row[f]))}</span></div>'
                          f'<div class="strack"><div class="smid"></div><div class="sbar" style="{side};background:{color}"></div></div>'
                          f'<div class="sval num" style="color:{color}">{"+" if v > 0 else "−"}{fr(abs(v), 2)}</div></div>')
        st.markdown(f'<div class="slegend"><span><i style="background:{ORANGE}"></i>Augmente le risque</span>'
                    f'<span><i style="background:{BLUE}"></i>Réduit le risque</span></div>{rows_html}',
                    unsafe_allow_html=True)

# =====================================================================
with tab_camp:
    n_test = report["test_size"]["customers"]
    st.markdown(f"<div class='lead' style='margin-top:.6rem'>Simulation sur les {fr(n_test, 0)} clients du jeu de test, "
                "jamais vus par le modèle. Réglez les hypothèses de l'offre de rétention : la valeur nette et le seuil "
                "optimal se recalculent.</div>", unsafe_allow_html=True)
    with st.container(key="box1"):
        st.markdown("<h3>Hypothèses de la campagne</h3>", unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        offer = h1.select_slider("Coût de l'offre (mois d'abonnement offerts)", [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                                 value=float(c["offer_months"]), format_func=lambda v: f"{fr(v, 1)} mois")
        success = h2.slider("Taux de succès de l'offre sur un churner", 5, 80, int(round(100 * c["success_rate"])), 5,
                            format="%d %%") / 100
        kept = h3.slider("Mois de revenus conservés si retenu", 3, 24, int(c["retained_months"]), 1, format="%d mois")
    cv = camp.assign(value=lambda d: success * kept * d["monthly_churners_targeted"] - offer * d["monthly_targeted"])
    best = cv.loc[cv["value"].idxmax()]
    sel = st.slider("Seuil de ciblage (probabilité de churn)", 0, 100, int(round(100 * meta["threshold"])), 1,
                    format="%d %%") / 100
    st.caption("Par défaut, le seuil du modèle, choisi sur les données d'entraînement. L'optimum calculé sur le test "
               "n'est affiché que pour information : le choisir sur le test serait de la triche.")
    cur = cv.loc[(cv["threshold"] - sel).abs().idxmin()]
    total_churn = int(cv.loc[0, "churners_targeted"])
    k = st.columns(4, gap="small")
    cards = [("hero", "Valeur nette de la campagne", money(cur["value"]), f"optimum sur le test : {money(best['value'])} à {pct(best['threshold'], 0)}"),
             ("card", "Clients ciblés", fr(cur["targeted"], 0), f"{pct(cur['targeted'] / n_test, 0)} de la base"),
             ("card", "Churners atteints", f"{fr(cur['churners_targeted'], 0)} / {fr(total_churn, 0)}", f"{pct(cur['churners_targeted'] / total_churn, 0)} des départs"),
             ("card", "Cibler tout le monde", money(cv.loc[0, 'value']), "pour comparaison")]
    for col, (kind, lab, val, sub) in zip(k, cards):
        sub_style = "color:#FFE9DA" if kind == "hero" else ""
        col.markdown(f"""<div class="{kind}"><div class="label" style="margin-top:.2rem">{lab}</div>
        <div class="value num">{val}</div><div class="sub" style="{sub_style}">{sub}</div></div>""",
                     unsafe_allow_html=True)
    with st.container(key="box2"):
        st.markdown("<h3>Valeur nette selon le seuil</h3>", unsafe_allow_html=True)
        st.caption("Gain = churners ciblés × taux de succès × mois conservés × facture ; coût = clients ciblés × offre × facture.")
        base_chart = alt.Chart(cv).mark_line(color=ORANGE, strokeWidth=2.5).encode(
            x=alt.X("threshold:Q", title="Seuil de ciblage",
                    axis=alt.Axis(labelExpr="format(datum.value * 100, '.0f') + ' %'")),
            y=alt.Y("value:Q", title="Valeur nette ($)",
                    axis=alt.Axis(labelExpr="replace(format(datum.value, ',.0f'), /,/g, ' ')")),
            tooltip=[alt.Tooltip("threshold:Q", title="Seuil", format=".0%"),
                     alt.Tooltip("value:Q", title="Valeur nette ($)", format=",.0f"),
                     alt.Tooltip("targeted:Q", title="Clients ciblés")])
        rule = alt.Chart(pd.DataFrame({"threshold": [sel]})).mark_rule(color=INK, strokeDash=[4, 4]).encode(x="threshold:Q")
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=GREY).encode(y="y:Q")
        st.altair_chart((zero + base_chart + rule).properties(height=300).configure_view(strokeWidth=0)
                        .configure(background="transparent", font="Montserrat, system-ui, sans-serif")
                        .configure_axis(labelFontSize=14, titleFontSize=14, titleFontWeight=500, labelColor=MUTED,
                                        titleColor=MUTED, gridColor="#ECEBE8", domainColor="#D6D5D1"),
                        use_container_width=True)

# =====================================================================
with tab_perf:
    tm, td = report["test_metrics"], report["test_decision"]
    st.markdown(f"<div class='lead' style='margin-top:.6rem'>Mesuré sur {fr(n_test, 0)} clients mis de côté avant "
                "l'entraînement.</div>", unsafe_allow_html=True)
    k = st.columns(4, gap="small")
    cards = [("hero", "AUC sur le test", fr(tm["roc_auc"], 3), f"PR-AUC {fr(tm['pr_auc'], 2)}"),
             ("card", "Lift des 20 % les plus risqués", f"× {fr(tm['lift_top20'], 2)}", f"{pct(tm['capture_top20'], 0)} des départs captés"),
             ("card", "Départs détectés au seuil", pct(td["recall_churn"], 0), f"précision {pct(td['precision_churn'], 0)}"),
             ("card", "Taux de churn de la base", pct(report["cleaning"]["churn_rate"]), "7 043 clients")]
    for col, (kind, lab, val, sub) in zip(k, cards):
        sub_style = "color:#FFE9DA" if kind == "hero" else ""
        col.markdown(f"""<div class="{kind}"><div class="label" style="margin-top:.2rem">{lab}</div>
        <div class="value num">{val}</div><div class="sub" style="{sub_style}">{sub}</div></div>""",
                     unsafe_allow_html=True)
    st.write("")
    a, b = st.columns([1.1, 1], gap="medium")
    with a:
        with st.container(key="box3"):
            st.markdown("<h3>Neuf modèles comparés</h3>", unsafe_allow_html=True)
            st.caption("AUC moyenne en validation croisée sur 5 plis. En orange, le modèle retenu.")
            comp = pd.DataFrame(report["comparison"])
            comp["Modèle"] = comp["model"].map(MODEL_LABEL)
            comp["debut"] = 0.7
            comp["auc_fr"] = comp["cv_roc_auc"].map(lambda v: fr(v, 3))
            comp["role"] = np.where(comp["model"] == report["selected_model"], "Retenu", "Comparé")
            order = list(comp.sort_values("cv_roc_auc", ascending=False)["Modèle"])
            bars = alt.Chart(comp).mark_bar(cornerRadiusEnd=4, height=16).encode(
                x=alt.X("cv_roc_auc:Q", title="AUC (validation croisée)", scale=alt.Scale(domain=[0.7, 0.9])),
                x2="debut:Q",
                y=alt.Y("Modèle:N", sort=order, title=None, axis=alt.Axis(labelColor=INK, labelFontSize=14, ticks=False,
                                                                          labelLimit=220)),
                color=alt.Color("role:N", scale=alt.Scale(domain=["Retenu", "Comparé"], range=[ORANGE, GREY]), legend=None),
                tooltip=[alt.Tooltip("Modèle:N"), alt.Tooltip("cv_roc_auc:Q", title="AUC", format=".3f"),
                         alt.Tooltip("cv_pr_auc:Q", title="PR-AUC", format=".3f")])
            labels = bars.mark_text(align="left", dx=5, fontSize=14, color=INK).encode(text="auc_fr:N", color=alt.value(INK))
            st.altair_chart((bars + labels).properties(height=330, padding={"left": 5, "right": 40, "top": 5, "bottom": 5})
                            .configure_view(strokeWidth=0).configure(background="transparent", font="Montserrat, system-ui, sans-serif")
                            .configure_axis(labelFontSize=14, titleFontSize=14, labelColor=MUTED, titleColor=MUTED,
                                            gridColor="#ECEBE8", domainColor="#D6D5D1"), use_container_width=True)
    with b:
        with st.container(key="box4"):
            st.markdown("<h3>Taux de churn par décile de score</h3>", unsafe_allow_html=True)
            st.caption("Taux de départ observé (en %) des clients du test, des plus risqués (1) aux moins risqués (10).")
            dec = pd.DataFrame(report["deciles"])
            ramp = ORANGE_RAMP
            cells = ""
            for d, v in zip(dec["decile"], dec["taux_churn_observe"]):
                i = min(int(v * len(ramp) / 0.8), len(ramp) - 1)
                fg = "#fff" if i >= 4 else INK
                cells += (f'<div class="dcol"><div class="rowlab">{int(d)}</div><div class="cell num" '
                          f'style="background:{ramp[i]};color:{fg}" title="Décile {int(d)} : {pct(v)}">{fr(100 * v, 0)}</div></div>')
            st.markdown(f'<div class="grid">{cells}</div>', unsafe_allow_html=True)
            st.markdown(f"<div class='sub' style='margin-top:1rem'>Les 20 % de clients les plus risqués concentrent "
                        f"<b style='color:#1E2A3B'>{pct(tm['capture_top20'], 0)}</b> des départs.</div>", unsafe_allow_html=True)
            by_c = report["churn_by_contract"]
            st.markdown("<h3 style='margin-top:1.2rem'>Churn par type de contrat</h3>", unsafe_allow_html=True)
            st.markdown("".join(f"<div class='srow'><div class='slab'>{CONTRACT[k]}</div><div class='strack'>"
                                f"<div class='sbar' style='left:0;width:{v / 0.5 * 100:.0f}%;background:{ORANGE}'></div></div>"
                                f"<div class='sval num'>{pct(v, 0)}</div></div>" for k, v in by_c.items()),
                        unsafe_allow_html=True)

# =====================================================================
with tab_method:
    st.markdown(f"""<div class="card" style="margin-top:.6rem">
<h3 style="font-size:1.15rem;font-weight:500">Comment le modèle a été construit</h3>
<div style="max-width:72ch;color:#3A4556;font-size:.95rem;line-height:1.65">
<p><b>Données.</b> 7 043 abonnés d'un opérateur fictif (jeu d'exemple IBM, publié sur Kaggle), 26,5 % de résiliations.
Les 11 clients arrivés dans le mois n'avaient pas encore de total facturé : il est mis à 0.</p>
<p><b>Comparaison.</b> Les neuf modèles du sujet (régression logistique, SVM, KNN, Naive Bayes, Random Forest, Gradient
Boosting, XGBoost, LightGBM, CatBoost) sont évalués en validation croisée sur 80 % des données. Le modèle retenu est le
meilleur parmi ceux dont on sait calculer les explications exactes ; ici, c'est aussi le meilleur tout court.</p>
<p><b>Décision.</b> Le seuil de ciblage maximise la valeur d'une campagne de rétention simulée (offre d'un mois, 30 %
de succès sur un churner, 12 mois de revenus conservés). Il est choisi hors du jeu de test, et les hypothèses sont
réglables dans l'onglet Campagne.</p>
<p><b>Explications.</b> Chaque score est décomposé en valeurs SHAP exactes, regroupées sur les variables d'origine.</p>
<p><b>Mise en production.</b> API FastAPI (clé API, contrôle de cohérence des services, intégrité du modèle), conteneur
Docker sans droits administrateur, tests et intégration continue.</p>
<p><b>Limites.</b> Données d'exemple d'un seul mois, sans historique : le modèle ne dit pas <i>quand</i> le client
partira. L'ancienneté et le total facturé sont très liés : leurs contributions SHAP se partagent l'effet. Les hypothèses
de campagne sont illustratives.</p>
<p><a href="{GITHUB_URL}" target="_blank" rel="noopener" style="color:#1E2A3B;font-weight:600;border-bottom:1.5px solid #F4721E;text-decoration:none">Lire le code et le rapport complet sur GitHub</a></p>
</div></div>""", unsafe_allow_html=True)

footer()
