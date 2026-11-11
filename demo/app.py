"""Tableau de bord Churn télécom : navigation, en-tête et filtres communs.

Pilotage   : Synthèse (accueil)
Activité   : Clients et revenu · Services · Churn par segment
Rétention  : Clients à risque · Modèle et campagne

Lancer : streamlit run demo/app.py
"""
import streamlit as st

from common import apply_styles, hero, sidebar_filters

st.set_page_config(page_title="Churn télécom", page_icon="📶", layout="wide", initial_sidebar_state="auto")
apply_styles()

PAGES = {
    "Pilotage": [
        st.Page("views/overview.py", title="Synthèse", icon=":material/dashboard:", default=True),
    ],
    "Activité": [
        st.Page("views/customers.py", title="Clients et revenu", icon=":material/groups:"),
        st.Page("views/services.py", title="Services", icon=":material/router:"),
        st.Page("views/segments.py", title="Churn par segment", icon=":material/grid_view:"),
    ],
    "Rétention": [
        st.Page("views/risk.py", title="Clients à risque", icon=":material/warning:"),
        st.Page("views/model.py", title="Modèle et campagne", icon=":material/psychology:"),
    ],
}
SUBTITLES = {
    "Synthèse": "Vue d'ensemble pour la direction et le marketing : taille de la base, départs, revenu et clients à "
                "retenir. Tout est interactif : cliquez sur un graphique ou utilisez les filtres de gauche.",
    "Clients et revenu": "Qui sont les abonnés, combien ils rapportent chaque mois et quelle part de ce revenu part avec "
                         "les clients qui résilient.",
    "Services": "Téléphonie, accès Internet et options : ce que les clients ont souscrit et le lien avec les départs.",
    "Churn par segment": "Où se concentrent les départs : grilles croisées par contrat, ancienneté, paiement et accès "
                         "Internet.",
    "Clients à risque": "Les clients encore actifs classés par le modèle : combien sont à cibler, combien ils "
                        "rapportent et où ils se trouvent.",
    "Modèle et campagne": "Décrivez un abonné : le modèle estime sa probabilité de résilier, indique s'il faut lui "
                          "proposer l'offre de rétention et explique pourquoi. Entraîné sur 7 043 clients.",
}

page = st.navigation(PAGES, position="sidebar")
group = next(g for g, items in PAGES.items() for p in items if p.title == page.title)

if page.title != "Modèle et campagne":
    sidebar_filters()
    with st.sidebar:
        st.markdown("<div class='sbnote'>Données d'exemple d'un opérateur fictif (IBM / Kaggle), photo de la base à un "
                    "instant donné. Les segments de moins de 5 clients sont regroupés pour l'anonymat.</div>",
                    unsafe_allow_html=True)

hero(f"{group} · Churn télécom", page.title, SUBTITLES[page.title])
page.run()
