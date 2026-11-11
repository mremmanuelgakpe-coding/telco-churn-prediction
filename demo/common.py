"""Socle commun des deux vues : palette, styles, formats français, chargement sécurisé des données."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

GITHUB_URL = "https://github.com/mremmanuelgakpe-coding/telco-churn-prediction"
LINKEDIN_URL = "https://www.linkedin.com/in/egakpe"
CONTACT_MAIL = "mailto:mremmanuelgakpe@icloud.com?subject=Portfolio%20Data%20%26%20IA%20-%20Churn%20t%C3%A9l%C3%A9com"
LINKS = (f'<a href="{LINKEDIN_URL}" target="_blank" rel="noopener noreferrer">LinkedIn</a> · '
         f'<a href="{CONTACT_MAIL}">Me contacter</a> · '
         f'<a href="{GITHUB_URL}" target="_blank" rel="noopener noreferrer">Code</a>')

# Palette de marque : crème, bleu, orange, avec du blanc pour les cartes
CREAM, BLUE, ORANGE, WHITE = "#E9D9CA", "#4D6080", "#F4721E", "#FFFFFF"
INK, MUTED, GRID = "#1E2A3B", "#5E6878", "#ECE4DB"
DATA_BLUE = "#3B5C99"  # bleu de données : même teinte, plus saturé (lisible dans les graphiques)
ORANGE_RAMP = ["#FBF4EE", "#FDE3D0", "#FBC9A6", "#F8A26B", "#F4721E", "#C9560F"]
CHART_FONT = "Montserrat, system-ui, sans-serif"

CONTRACT = {"Month-to-month": "Sans engagement", "One year": "Engagement 1 an", "Two year": "Engagement 2 ans"}
CONTRACT_SHORT = {"Month-to-month": "Sans eng.", "One year": "1 an", "Two year": "2 ans"}
INTERNET = {"Fiber optic": "Fibre", "DSL": "ADSL", "No": "Sans Internet"}
PAYMENT = {"Electronic check": "Chèque électronique", "Mailed check": "Chèque papier",
           "Bank transfer (automatic)": "Prélèvement bancaire", "Credit card (automatic)": "Carte bancaire (auto)"}
GROUPED = "Regroupé"


def fr(x, nd=1):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


def pct(x, nd=1):
    return f"{fr(100 * x, nd)} %"


def money(x):
    return f"{x:,.0f} $".replace(",", " ")


@st.cache_resource
def load_model():
    path = ROOT / "models" / "model.joblib"
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != meta["model_sha256"]:
        st.error("Le fichier du modèle ne correspond pas à son empreinte. Chargement interrompu.")
        st.stop()
    report = json.loads((ROOT / "reports" / "metrics.json").read_text())
    camp = pd.read_csv(ROOT / "reports" / "campaign_table_test.csv")
    return joblib.load(path), meta, report, camp


@st.cache_data
def load_activity():
    d = ROOT / "reports" / "activity"
    return (pd.read_csv(d / "activity_cube.csv"), pd.read_csv(d / "tenure_curve.csv"),
            json.loads((d / "manifest.json").read_text()))


def header(title: str, lead: str):
    from datetime import date

    st.markdown(f"""
<div class="topbar"><div class="brand"><div class="brand-mark"></div>Churn télécom</div>
<div class="author">Projet portfolio de <b>Emmanuel</b> · {LINKS}</div></div>
<div class="title-row"><h1>{title}</h1><span class="chip">{date.today().strftime('%d/%m/%Y')}</span></div>
<div class="lead">{lead}</div>""", unsafe_allow_html=True)


def kpi(col, label, value, sub="", kind="card"):
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    col.markdown(f'<div class="{kind}"><div class="label">{label}</div><div class="value num">{value}</div>{sub_html}</div>',
                 unsafe_allow_html=True)


def chart_config(chart, height):
    return (chart.properties(height=height).configure_view(strokeWidth=0)
            .configure(background="transparent", font=CHART_FONT)
            .configure_axis(labelFontSize=14, titleFontSize=14, titleFontWeight=500, labelColor=MUTED,
                            titleColor=MUTED, gridColor=GRID, domainColor="#D9CFC4", tickColor="#D9CFC4")
            .configure_legend(labelFontSize=14, titleFontSize=14, labelColor=INK, titleColor=MUTED))


def footer():
    st.markdown("<div class='foot'>Démonstration pédagogique sur données publiques d'exemple (IBM / Kaggle). "
                "Aucune donnée saisie n'est enregistrée. Une question, une mission ou un poste à pourvoir ? "
                f"Échangeons sur <a href='{LINKEDIN_URL}' target='_blank' rel='noopener noreferrer'>LinkedIn</a> "
                f"ou <a href='{CONTACT_MAIL}'>par e-mail</a>.</div>", unsafe_allow_html=True)


CSS = f"""
@font-face {{ font-family:'Montserrat'; src:url('app/static/fonts/montserrat-latin-wght-normal.woff2') format('woff2');
             font-weight:100 900; font-style:normal; font-display:swap; }}
@font-face {{ font-family:'Montserrat'; src:url('app/static/fonts/montserrat-latin-wght-italic.woff2') format('woff2');
             font-weight:100 900; font-style:italic; font-display:swap; }}
:root {{ --page:{CREAM}; --panel:#F7F1EA; --card:{WHITE}; --ink:{INK}; --muted:{MUTED}; --orange:{ORANGE};
        --blue:{BLUE}; --blue-deep:#3A4A66; --tint:#FDE3D0; --line:#EFE7DE; }}
.stApp, .stApp p, .stApp label, .stApp input, .stApp textarea, .stApp button {{ font-family:'Montserrat', system-ui, sans-serif; }}
[data-testid="stTooltipIcon"] button {{ background:transparent !important; color:var(--muted) !important; }}
.stApp {{ background:var(--page); color:var(--ink); }}
header[data-testid="stHeader"] {{ background:var(--page); }}
.block-container {{ max-width:1280px; padding:1.4rem 1.6rem 3rem; background:var(--panel); border-radius:30px;
                    margin-top:1.2rem; margin-bottom:1.5rem; }}
h1, h2, h3 {{ font-family:'Montserrat', sans-serif !important; color:var(--ink); letter-spacing:-0.02em; }}
.num {{ font-variant-numeric:tabular-nums; }}

/* navigation entre les vues (barre du haut) */
header[data-testid="stHeader"] a[data-testid="stTopNavLink"] {{ border-radius:999px; padding:.35rem 1rem; }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"] span {{ font-size:.95rem; font-weight:600; color:var(--blue); }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"][aria-current="page"] {{ background:var(--card); }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"][aria-current="page"] span {{ color:var(--orange); }}

.topbar {{ display:flex; align-items:center; justify-content:space-between; gap:1rem; flex-wrap:wrap; }}
.brand {{ display:flex; align-items:center; gap:.5rem; font-weight:700; font-size:1.15rem; letter-spacing:-0.03em; color:var(--blue); }}
.brand-mark {{ width:26px; height:26px; border-radius:8px; background:var(--blue); position:relative; }}
.brand-mark::after {{ content:""; position:absolute; right:-4px; bottom:-4px; width:12px; height:12px; border-radius:50%;
                      background:var(--orange); border:3px solid var(--panel); }}
.author {{ color:var(--muted); font-size:.875rem; }}
.author a {{ color:var(--ink); font-weight:600; text-decoration:none; border-bottom:1.5px solid var(--orange); }}
.title-row {{ display:flex; align-items:center; gap:.75rem; margin:1.1rem 0 .3rem; flex-wrap:wrap; }}
.title-row h1 {{ font-size:1.9rem; font-weight:600; margin:0; padding:0; }}
.chip {{ background:var(--card); border-radius:999px; padding:.25rem .7rem; font-size:.875rem; color:var(--muted); }}
.lead {{ color:var(--muted); font-size:.92rem; max-width:75ch; margin-bottom:.6rem; }}

.stTabs [role="tablist"] {{ gap:.25rem; background:var(--card); border-radius:999px; padding:.3rem; width:fit-content;
                            max-width:100%; overflow-x:auto; border:none !important; box-shadow:none !important; }}
.stTabs [role="tab"] {{ border-radius:999px; padding:.4rem 1rem; color:var(--muted); border:none !important; }}
.stTabs [role="tab"] p {{ font-size:.875rem; font-weight:500; }}
.stTabs [role="tab"][data-selected="true"] {{ background:var(--panel); }}
.stTabs [role="tab"][data-selected="true"] p {{ color:var(--orange); font-weight:600; }}
.stTabs [role="tab"]:focus-visible {{ outline:3px solid var(--orange); outline-offset:1px; }}
.stTabs .react-aria-SelectionIndicator {{ display:none !important; }}

.card {{ background:var(--card); border-radius:22px; padding:1.05rem 1.2rem; height:100%; margin-bottom:12px; }}
.card .label, .hero .label, .bluecard .label {{ font-size:.92rem; font-weight:500; }}
.card .label {{ color:var(--ink); }}
.card .value, .bluecard .value {{ font-size:1.8rem; font-weight:600; letter-spacing:-0.03em; margin-top:.25rem; }}
.card .sub {{ font-size:.875rem; color:var(--muted); margin-top:.2rem; }}
.hero {{ background:radial-gradient(120% 120% at 100% 0%, #FFA266 0%, {ORANGE} 45%, #C4520F 100%); color:#fff;
         border-radius:22px; padding:1.05rem 1.2rem; height:100%; margin-bottom:12px; position:relative; overflow:hidden; }}
.hero .label {{ color:#FFF1E6; }}
.hero .value {{ font-size:2.4rem; font-weight:600; letter-spacing:-0.04em; line-height:1.1; margin-top:.2rem; }}
.hero .sub {{ color:#FFF1E6; font-size:.875rem; margin-top:.2rem; }}
.bluecard {{ background:var(--blue); color:#fff; border-radius:22px; padding:1.05rem 1.2rem; height:100%; margin-bottom:12px; }}
.bluecard .label, .bluecard .sub {{ color:#E3E8F1; }}
.bluecard .sub {{ font-size:.875rem; margin-top:.2rem; }}
.icon {{ width:26px; height:26px; border-radius:7px; background:var(--blue); color:#fff; display:inline-flex;
         align-items:center; justify-content:center; font-size:.875rem; font-weight:700; margin-bottom:.7rem; }}
.hero .icon {{ background:#fff; color:var(--orange); }}
.badge {{ position:absolute; top:1rem; right:1rem; background:rgba(255,255,255,.22); border-radius:999px; padding:.15rem .6rem;
          font-size:.875rem; }}
.badge.light {{ background:var(--panel); color:var(--muted); }}
.decision {{ display:inline-block; border-radius:999px; padding:.2rem .8rem; font-weight:600; font-size:1.1rem; margin-top:.3rem; }}
.decision.ok {{ background:#E7ECF4; color:var(--blue); }}
.decision.ko {{ background:var(--tint); color:#B2470A; }}

.scale {{ display:flex; gap:6px; margin-top:.8rem; }}
.seg {{ flex:1; height:46px; border-radius:12px; position:relative; }}
.seg.on {{ outline:3px solid var(--blue); outline-offset:2px; }}
.seg span {{ position:absolute; bottom:-1.6rem; left:0; right:0; text-align:center; font-size:.875rem; color:var(--muted); }}
.scale-top {{ display:flex; justify-content:space-between; font-size:.875rem; color:var(--muted); }}
.scale-top b {{ color:var(--ink); font-size:1.2rem; font-weight:600; }}

.dark {{ background:linear-gradient(160deg, var(--blue) 0%, var(--blue-deep) 100%); color:#F4F1EC; border-radius:24px;
         padding:1.3rem 1.3rem 1rem; }}
.dark h3 {{ color:#fff; font-size:1.15rem; font-weight:500; margin:0 0 .2rem; }}
.dark .hint {{ color:#D4DBE7; font-size:.875rem; margin-bottom:.8rem; }}
.dark .count {{ float:right; background:var(--orange); color:#fff; border-radius:999px; padding:.05rem .6rem; font-size:.875rem; font-weight:700; }}
.reason {{ display:flex; justify-content:space-between; align-items:center; gap:.8rem; padding:.55rem 0;
           border-top:1px solid rgba(255,255,255,.16); }}
.reason .t {{ font-size:.9rem; color:#fff; }}
.reason .v {{ font-size:.875rem; color:#D4DBE7; }}
.dot {{ width:24px; height:24px; flex:0 0 24px; border-radius:50%; display:flex; align-items:center; justify-content:center;
        font-size:.875rem; font-weight:700; }}
.dot.up {{ background:var(--orange); color:#fff; }}
.dot.down {{ background:#fff; color:var(--blue); }}
.group {{ font-size:.875rem; color:#D4DBE7; margin:.7rem 0 .1rem; }}

[class*="st-key-box"], .st-key-form, .st-key-chart {{ background:var(--card); border-radius:22px; padding:1.1rem 1.2rem 1.2rem; }}
[class*="st-key-box"] h3, .st-key-form h3, .st-key-chart h3 {{ font-size:1.1rem; font-weight:600; margin:0; }}
.st-key-chart, [class*="st-key-box"] {{ margin-top:12px; }}
div[data-baseweb="select"] > div, .stNumberInput input {{ border-radius:12px !important; background:#FBF8F5 !important; }}
.stButton button, [data-testid="stPageLink"] a {{ border-radius:999px; border:none; background:var(--panel); color:var(--ink);
                                                  font-weight:600; font-size:.875rem; }}
.stButton button:hover {{ background:var(--tint); color:#B2470A; }}
.stButton button:focus-visible, [data-testid="stPageLink"] a:focus-visible {{ outline:3px solid var(--orange); outline-offset:2px; }}
.st-key-goto a, .st-key-goto2 a {{ background:var(--blue) !important; padding:.45rem 1.1rem !important; width:fit-content; max-width:100%; height:auto; }}
.st-key-goto a p, .st-key-goto2 a p {{ color:#fff !important; font-weight:600; white-space:normal !important; overflow:visible !important; text-overflow:clip !important; }}

[data-testid="stButtonGroup"] button[aria-checked="false"] {{ background:#FFFFFF; color:var(--ink); border-color:#E6DCD2; }}
[data-testid="stButtonGroup"] button[aria-checked="false"]:hover {{ background:var(--panel); color:var(--ink); }}
.slegend {{ display:flex; gap:1.2rem; font-size:.875rem; color:var(--muted); margin:.4rem 0 .6rem; flex-wrap:wrap; }}
.slegend i {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:.4rem; vertical-align:-1px; }}
.srow {{ display:grid; grid-template-columns:minmax(170px, 1.1fr) 2fr 70px; gap:.9rem; align-items:center; padding:.45rem 0;
         border-top:1px solid var(--line); }}
.slab {{ font-size:.9rem; color:var(--ink); }}
.slab span {{ display:block; font-size:.875rem; color:var(--muted); }}
.strack {{ position:relative; height:14px; background:#F6F0EA; border-radius:4px; }}
.strack.center {{ background:transparent; }}
.smid {{ position:absolute; left:50%; top:-6px; bottom:-6px; width:1.5px; background:#D9CFC4; }}
.sbar {{ position:absolute; top:0; height:14px; border-radius:4px; min-width:2px; }}
.sval {{ font-size:.875rem; font-weight:600; text-align:right; }}

.grid {{ display:grid; grid-template-columns:repeat(10, 1fr); gap:6px; margin-top:.8rem; }}
.dcol {{ display:flex; flex-direction:column; align-items:center; gap:4px; }}
.dcol .cell {{ width:100%; }}
.cell {{ height:40px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:.875rem; font-weight:600; }}
.rowlab {{ font-size:.875rem; color:var(--muted); text-align:center; }}
.note {{ background:#FBF4EE; border-radius:14px; padding:.6rem .9rem; font-size:.875rem; color:var(--muted); margin:.4rem 0; }}
.foot {{ color:var(--muted); font-size:.875rem; margin-top:1.4rem; }}
@media (max-width: 640px) {{
  .block-container {{ padding:1rem .9rem 2rem; border-radius:22px; margin-top:.6rem; }}
  .hero .value {{ font-size:2rem; }} .card .value, .bluecard .value {{ font-size:1.5rem; }}
  .grid {{ grid-template-columns:repeat(5, 1fr); gap:8px; }}
  .srow {{ grid-template-columns:1fr 64px; gap:.3rem .6rem; }}
  .srow .strack {{ grid-column:1 / -1; grid-row:2; }}
}}
"""


def apply_styles():
    st.session_state["_card_seq"] = 0
    st.markdown(f"<style>{CSS}{CSS_DASH}</style>", unsafe_allow_html=True)


# =====================================================================================
# Tableau de bord Activité : filtres, cartes KPI de même taille, cartes, donuts, heatmaps
# =====================================================================================

import altair as alt  # noqa: E402

DATA_BLUE_LIGHT = "#9DB0D1"
NEUTRAL = "#CFC4B8"
SENIOR = {"Non senior": "Moins de 65 ans", "Senior": "65 ans et plus"}
TENURE = ["0-6 mois", "7-12 mois", "13-24 mois", "25-48 mois", "49-72 mois"]
RISK = ["Faible", "Modéré", "Élevé", "Très élevé"]
RISK_COLORS = dict(zip(RISK, ["#FBC9A6", "#F8A26B", "#F4721E", "#B54A0B"]))
FILTERS = {  # dimension du cube : (libellé, valeurs dans l'ordre d'affichage, libellés français)
    "contract": ("Contrat", list(CONTRACT), CONTRACT),
    "internet": ("Accès Internet", list(INTERNET), INTERNET),
    "payment": ("Mode de paiement", list(PAYMENT), PAYMENT),
    "senior": ("Âge", list(SENIOR), SENIOR),
    "tenure_band": ("Ancienneté", TENURE, {}),
    "risk_band": ("Niveau de risque", RISK, {}),
}


def lab(dim, value):
    return FILTERS[dim][2].get(value, value)


def reset_filters():
    for k in FILTERS:
        st.session_state[f"flt_{k}"] = []


def sidebar_filters():
    with st.sidebar:
        with st.container(key="sidebar-filters"):
            st.markdown("<div class='sbtitle'>Filtres</div>", unsafe_allow_html=True)
            for k, (label, values, names) in FILTERS.items():
                st.multiselect(label, values, key=f"flt_{k}", format_func=lambda v, n=names: n.get(v, v),
                               placeholder="Tous")
            if any(st.session_state.get(f"flt_{k}") for k in FILTERS):
                st.button("Réinitialiser les filtres", on_click=reset_filters, width="stretch")


def filtered(exclude=()):
    """Cube filtré. ``exclude`` : dimensions dont le filtre est ignoré (le graphique de cette dimension garde toutes
    ses modalités et met en avant celles choisies). Renvoie (cube filtré, clients exclus car regroupés, filtres)."""
    cube, _, _ = load_activity()
    exclude = (exclude,) if isinstance(exclude, str) else tuple(exclude)
    sel = {k: st.session_state.get(f"flt_{k}") or [] for k in FILTERS}
    use = {k: v for k, v in sel.items() if v and k not in exclude}
    d = cube[cube["contract"].isin(use["contract"])] if "contract" in use else cube
    excluded = 0
    if any(k != "contract" for k in use):
        excluded = int(d.loc[d["internet"] == GROUPED, "customers"].sum())
        d = d[d["internet"] != GROUPED]
        for k, v in use.items():
            if k != "contract":
                d = d[d[k].isin(v)]
    return d, excluded, sel


def view(*dims):
    """Données d'un graphique portant sur ``dims`` : tous les filtres sauf ceux de ces dimensions."""
    return filtered(exclude=dims)[0]


def _selected():
    return {k: st.session_state.get(f"flt_{k}") or [] for k in FILTERS}


def _on_pick(key, dims):
    """Clic sur un graphique : filtre tout le tableau de bord ; un second clic sur la même modalité annule."""
    event = st.session_state.get(key) or {}
    rows = (event.get("selection") or {}).get("pick") or []
    if not rows:
        return
    values = {dim: rows[0].get(dim) for dim in dims}
    if any(v is None or v not in FILTERS[dim][1] for dim, v in values.items()):
        return
    same = all((st.session_state.get(f"flt_{dim}") or []) == [v] for dim, v in values.items())
    for dim, v in values.items():
        st.session_state[f"flt_{dim}"] = [] if same else [v]


def _remove(dim, value):
    st.session_state[f"flt_{dim}"] = [v for v in st.session_state.get(f"flt_{dim}") or [] if v != value]


def _on(df, dims):
    """1 si la ligne fait partie de la sélection (ou si rien n'est sélectionné), 0 sinon : sert à l'opacité."""
    sel = _selected()
    on = pd.Series(True, index=df.index)
    for dim in dims:
        if sel[dim]:
            on &= df[dim].isin(sel[dim])
    return on.astype(int)


def clickable(parent, chart, dims, key, height):
    """Affiche un graphique Altair dont le clic filtre le tableau de bord sur ``dims``."""
    from functools import partial
    chart = chart_config(chart, height).configure_mark(cursor="pointer")
    parent.altair_chart(chart, use_container_width=True, key=key, on_select=partial(_on_pick, key, list(dims)),
                        selection_mode="pick")


def scope_note(d, excluded, sel):
    active = [(k, v) for k, vals in sel.items() for v in vals]
    if active:
        with st.container(horizontal=True, gap="small", key="chips"):
            st.markdown("<span class='chips-lab'>Filtres actifs</span>", unsafe_allow_html=True)
            for k, v in active:
                st.button(f"{FILTERS[k][0]} : {lab(k, v)}", key=f"chip_{k}_{v}", icon=":material/close:",
                          on_click=_remove, args=(k, v), help="Retirer ce filtre")
            st.button("Tout effacer", key="chip_clear", icon=":material/filter_alt_off:", on_click=reset_filters)
    else:
        st.markdown("<div class='tip'><b>Tableau de bord interactif</b> : cliquez sur une part de donut, une barre ou "
                    "une case de grille pour filtrer toutes les pages. Cliquez à nouveau pour annuler.</div>",
                    unsafe_allow_html=True)
    if d.empty:
        st.info("Aucun client pour cette combinaison de filtres. Retirez un filtre.")
        st.stop()
    if excluded:
        st.markdown(f"<div class='note'>{fr(excluded, 0)} clients de très petits segments (moins de 5 clients, regroupés "
                    "pour l'anonymat) ne sont pas comptés avec ces filtres.</div>", unsafe_allow_html=True)


def hero(eyebrow, title, subtitle):
    from datetime import date
    st.markdown(f"""<div class="herob"><div class="hb-top"><span class="eyebrow">{eyebrow}</span>
<span class="hb-meta">Projet portfolio de <b>Emmanuel</b> · {LINKS}
 · {date.today().strftime('%d/%m/%Y')}</span></div><h1>{title}</h1><p>{subtitle}</p></div>""", unsafe_allow_html=True)


def section(title, caption=None):
    cap = f"<div class='sec-cap'>{caption}</div>" if caption else ""
    st.markdown(f"<div class='sec'><span></span>{title}</div>{cap}", unsafe_allow_html=True)


def kpi_grid(items, cols=4):
    """Cartes KPI de taille identique. items : (libellé, valeur, sous-texte, ton) ; ton = '', 'orange', 'blue', 'hl'."""
    cards = ""
    for label, value, sub, *rest in items:
        tone = rest[0] if rest else ""
        cards += (f"<div class='kcard {tone}'><div class='k-label'>{label}</div>"
                  f"<div class='k-value num'>{value}</div><div class='k-sub'>{sub}</div></div>")
    st.markdown(f"<div class='kgrid c{cols}'>{cards}</div>", unsafe_allow_html=True)


def card(parent=None, title=None, caption=None):
    st.session_state["_card_seq"] = st.session_state.get("_card_seq", 0) + 1
    c = (parent or st).container(key=f"card-{st.session_state['_card_seq']}", height="stretch" if parent else "content")
    if title:
        cap = f"<div class='c-cap'>{caption}</div>" if caption else ""
        c.markdown(f"<div class='c-title'>{title}</div>{cap}", unsafe_allow_html=True)
    return c


def legend_rows(rows):
    """rows : (couleur, libellé, valeur affichée, part)."""
    out = "".join(f"<div class='dl'><i style='background:{c}'></i><span class='dl-n'>{n}</span>"
                  f"<span class='dl-v num'>{v}</span><span class='dl-p num'>{pct(p, 0)}</span></div>" for c, n, v, p in rows)
    return f"<div class='dleg'>{out}</div>"


def donut(parent, labels, values, colors, center_value, center_label, fmt=lambda v: fr(v, 0), height=230, total=None,
          dim=None, codes=None, key=None):
    df = pd.DataFrame({"cat": labels, "val": values, "order": range(len(labels))})
    df["share"] = df["val"] / df["val"].sum()
    df["val_fr"], df["share_fr"] = df["val"].map(fmt), df["share"].map(lambda v: pct(v, 1))
    if dim:
        df[dim] = codes
        df["on"] = _on(df, [dim])
    else:
        df["on"] = 1
    arc = alt.Chart(df).mark_arc(innerRadius=66, outerRadius=102, stroke="#FFFFFF", strokeWidth=3, cornerRadius=5).encode(
        theta=alt.Theta("val:Q", stack=True), order=alt.Order("order:Q"),
        color=alt.Color("cat:N", scale=alt.Scale(domain=labels, range=colors), legend=None),
        opacity=alt.Opacity("on:Q", scale=alt.Scale(domain=[0, 1], range=[0.25, 1]), legend=None),
        tooltip=[alt.Tooltip("cat:N", title="Segment"), alt.Tooltip("val_fr:N", title="Valeur"),
                 alt.Tooltip("share_fr:N", title="Part")])
    center = pd.DataFrame({"t": [center_value], "s": [center_label]})
    t1 = alt.Chart(center).mark_text(fontSize=22, fontWeight=700, color=INK, dy=-9, font=CHART_FONT).encode(text="t:N")
    t2 = alt.Chart(center).mark_text(fontSize=14, color=MUTED, dy=15, font=CHART_FONT).encode(text="s:N")
    if dim:
        arc = arc.add_params(alt.selection_point(name="pick", fields=[dim]))
        clickable(parent, alt.layer(arc, t1, t2), [dim], key or f"donut_{dim}", height)
    else:
        parent.altair_chart(chart_config(alt.layer(arc, t1, t2), height), use_container_width=True)
    rows = list(zip(colors, labels, df["val"].map(fmt), df["share"]))
    faded = [i for i, on in enumerate(df["on"]) if not on]
    html = legend_rows(rows)
    if faded:  # légende : modalités non sélectionnées en retrait
        parts = html.split("<div class='dl'>")
        html = parts[0] + "".join(("<div class='dl off'>" if i in faded else "<div class='dl'>") + p_
                                  for i, p_ in enumerate(parts[1:]))
    parent.markdown(html, unsafe_allow_html=True)
    if total and total > df["val"].sum():
        parent.markdown(f"<div class='c-cap'>Hors {fr(total - df['val'].sum(), 0)} clients de très petits segments, "
                        "regroupés pour l'anonymat.</div>", unsafe_allow_html=True)


BLUE_RAMP = ["#F1F4F9", "#C9D4E8", "#9DB0D1", "#6A85B5", "#3B5C99"]


def heatmap(parent, df, x, y, value, x_order, y_order, text, height=None, scale_max=None, ramp=None, dims=None,
            key=None):
    """Grille colorée (rampe orange) avec la valeur écrite dans chaque case."""
    df = df.copy()
    df["txt"] = text
    df["on"] = _on(df, dims) if dims else 1
    vmax = scale_max or float(df[value].max())
    rect = alt.Chart(df).mark_rect(cornerRadius=8, stroke="#FFFFFF", strokeWidth=4).encode(
        x=alt.X(f"{x}:N", sort=x_order, title=None, axis=alt.Axis(orient="top", labelAngle=0, labelLimit=160, ticks=False,
                                                                 domain=False, labelPadding=8)),
        y=alt.Y(f"{y}:N", sort=y_order, title=None, axis=alt.Axis(ticks=False, domain=False, labelLimit=180, labelPadding=8)),
        color=alt.Color(f"{value}:Q", scale=alt.Scale(domain=[0, vmax], range=ramp or ORANGE_RAMP[:5], clamp=True), legend=None),
        opacity=alt.Opacity("on:Q", scale=alt.Scale(domain=[0, 1], range=[0.22, 1]), legend=None),
        tooltip=[alt.Tooltip(f"{y}:N", title=" "), alt.Tooltip(f"{x}:N", title="  "), alt.Tooltip("txt:N", title="Valeur")])
    txt = alt.Chart(df).mark_text(fontSize=14, fontWeight=600, font=CHART_FONT).encode(
        x=alt.X(f"{x}:N", sort=x_order), y=alt.Y(f"{y}:N", sort=y_order), text="txt:N",
        color=alt.condition(alt.datum[value] > 0.62 * vmax, alt.value("#FFFFFF"), alt.value(INK)) if ramp == BLUE_RAMP
        else alt.value(INK))
    height = 52 * len(y_order) + 34
    if dims:
        from functools import partial
        rect = rect.add_params(alt.selection_point(name="pick", fields=list(dims)))
        key = key or "heat_" + "_".join(dims)
        chart = (chart_config(alt.layer(rect, txt), height).configure_axis(grid=False, labelFontSize=14, labelColor=INK)
                 .configure_mark(cursor="pointer"))
        parent.altair_chart(chart, use_container_width=True, key=key, on_select=partial(_on_pick, key, list(dims)),
                            selection_mode="pick")
        return
    parent.altair_chart(chart_config(alt.layer(rect, txt), height).configure_axis(grid=False, labelFontSize=14,
                        labelColor=INK), use_container_width=True)


def click_bars(parent, dim, table, sub=None, key=None, color=ORANGE, value="rate", fmt=lambda v: pct(v, 0)):
    """Barres horizontales Altair cliquables. ``table`` : index = codes de la dimension, colonnes ``value`` et
    ``customers`` ; ``sub`` : fonction (ligne) -> texte secondaire du libellé."""
    t = table.reset_index().rename(columns={table.index.name or "index": dim})
    t["name"] = t[dim].map(lambda v: lab(dim, v))
    if sub:
        t["name"] = [f"{n} · {sub(r)}" for n, (_, r) in zip(t["name"], t.iterrows())]
    t["val_fr"] = t[value].map(fmt)
    t["on"] = _on(t, [dim])
    order = list(t["name"])
    base = alt.Chart(t).encode(y=alt.Y("name:N", sort=order, title=None,
                                       axis=alt.Axis(ticks=False, domain=False, labelLimit=320, labelPadding=10,
                                                     labelFontSize=14, labelColor=INK)))
    vmax = float(t[value].max()) * 1.18 or 1
    bars = base.mark_bar(color=color, cornerRadiusEnd=5, height=18).encode(
        x=alt.X(f"{value}:Q", title=None, axis=None, scale=alt.Scale(domain=[0, vmax])),
        opacity=alt.Opacity("on:Q", scale=alt.Scale(domain=[0, 1], range=[0.25, 1]), legend=None),
        tooltip=[alt.Tooltip("name:N", title="Segment"), alt.Tooltip("val_fr:N", title="Valeur")])
    bars = bars.add_params(alt.selection_point(name="pick", fields=[dim]))
    text = base.mark_text(align="left", dx=6, fontSize=14, fontWeight=700, color=INK, font=CHART_FONT).encode(
        x=alt.X(f"{value}:Q"), text="val_fr:N")
    clickable(parent, alt.layer(bars, text), [dim], key or f"bars_{dim}", 44 * len(t) + 10)


def bar_rows(parent, rows, color=ORANGE, max_value=None, fmt=lambda v: pct(v, 0), wide=False):
    """Barres horizontales HTML. rows : (libellé, sous-texte, valeur)."""
    rows = list(rows)
    m = max_value or max([r[2] for r in rows] + [1e-9])
    out = ""
    for name, sub, v in rows:
        sub_html = f"<span>{sub}</span>" if sub else ""
        out += (f"<div class='srow{' wide' if wide else ''}'><div class='slab'>{name}{sub_html}</div><div class='strack'><div class='sbar' "
                f"style='left:0;width:{min(v / m, 1) * 100:.1f}%;background:{color}'></div></div>"
                f"<div class='sval num'>{fmt(v)}</div></div>")
    parent.markdown(out, unsafe_allow_html=True)


def insights(parent, items):
    parent.markdown("".join(f"<div class='ins'><span class='ins-dot'></span><div>{t}</div></div>" for t in items),
                    unsafe_allow_html=True)


def rate(d, col, num="churned", den="customers", order=None, names=None):
    g = d[d[col] != GROUPED].groupby(col)[[num, den, "mrr", "mrr_churned"]].sum()
    if order:
        g = g.reindex([o for o in order if o in g.index])
    g["rate"] = g[num] / g[den]
    if names:
        g.index = [names.get(i, i) for i in g.index]
    return g


CSS_DASH = """
[data-testid="stSidebar"] { background:linear-gradient(180deg, #4D6080 0%, #3D4D69 60%, #2F3B52 100%) !important; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top:.4rem; }
[data-testid="stSidebarNavLink"] { border-radius:999px !important; margin:2px 4px; padding:.35rem .8rem !important; }
[data-testid="stSidebarNavLink"] span { color:#E9EEF6 !important; font-size:.95rem !important; font-weight:500; }
[data-testid="stSidebarNavLink"]:hover { background:rgba(255,255,255,.10) !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] { background:#F4721E !important; }
[data-testid="stSidebarNavLink"][aria-current="page"] span { color:#FFFFFF !important; font-weight:700; }
[data-testid="stSidebarNavLink"] [data-testid="stIconMaterial"] { color:#FFFFFF !important; }
[data-testid="stNavSectionHeader"], [data-testid="stNavSectionHeader"] span { color:#FFC49C !important; font-size:.875rem !important;
  font-weight:700 !important; letter-spacing:.08em; text-transform:uppercase; }
[data-testid="stSidebarNavSeparator"] { border-color:rgba(255,255,255,.18) !important; }
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { display:none; }
.sbbrand { display:flex; align-items:center; gap:.6rem; color:#fff; font-weight:700; font-size:1.15rem; margin:.4rem .3rem 1rem; }
.sbbrand i { width:28px; height:28px; border-radius:9px; background:#fff; position:relative; display:inline-block; }
.sbbrand i::after { content:""; position:absolute; right:-4px; bottom:-4px; width:12px; height:12px; border-radius:50%;
  background:#F4721E; border:3px solid #4D6080; }
.sbbrand span { display:block; font-size:.875rem; font-weight:500; color:#D7DEEA; }
.st-key-sidebar-filters { background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.14); border-radius:20px;
  padding:14px 12px 10px; margin-top:.6rem; }
.sbtitle { color:#fff; font-weight:700; font-size:1rem; }
.st-key-sidebar-filters label p { color:#E9EEF6 !important; font-size:.875rem !important; font-weight:600; }
.st-key-sidebar-filters [data-testid="stMultiSelect"] div { background-color:#FFFFFF !important; }
.st-key-sidebar-filters [data-testid="stWidgetLabel"], .st-key-sidebar-filters [data-testid="stWidgetLabel"] div { background-color:transparent !important; }
.st-key-sidebar-filters [data-testid="stMultiSelect"] [data-baseweb="tag"] { background-color:#F4721E !important; }
.st-key-sidebar-filters [data-testid="stMultiSelect"] [data-baseweb="tag"] div { background-color:transparent !important; }
.st-key-sidebar-filters .stButton button { background:#FFFFFF; color:#3D4D69; }
[data-baseweb="tag"] { background:#F4721E !important; border-radius:999px !important; }
[data-baseweb="tag"] span { color:#fff !important; font-size:.875rem !important; }
.sbnote { color:#D7DEEA; font-size:.875rem; margin:.8rem .2rem 0; line-height:1.45; }

.herob { position:relative; overflow:hidden; border-radius:26px; padding:1.4rem 1.8rem 1.4rem; margin-bottom:1rem; color:#fff;
  background:radial-gradient(70% 140% at 100% 0%, rgba(244,114,30,.85) 0%, rgba(244,114,30,0) 60%),
             radial-gradient(50% 120% at 80% 130%, rgba(244,114,30,.45) 0%, rgba(244,114,30,0) 60%),
             linear-gradient(135deg, #4D6080 0%, #2F3B52 100%);
  box-shadow:0 16px 36px rgba(47,59,82,.22); }
.hb-top { display:flex; justify-content:space-between; gap:1rem; flex-wrap:wrap; align-items:center; }
.eyebrow { font-size:.875rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; color:#FFC49C; }
.hb-meta { font-size:.875rem; color:#E3E8F1; }
.foot a { color:#1E2A3B; font-weight:600; text-decoration:none; border-bottom:1.5px solid #F4721E; }
.hb-meta a { color:#fff; font-weight:600; text-decoration:none; border-bottom:1.5px solid #F4721E; }
.herob h1 { color:#fff !important; font-size:2rem; font-weight:700; margin:.5rem 0 .2rem; padding:0; letter-spacing:-.02em; }
.herob p { color:#E3E8F1; font-size:.95rem; margin:0; max-width:80ch; }

.sec { display:flex; align-items:center; gap:10px; font-weight:700; font-size:1.1rem; color:#1E2A3B; margin:1.1rem 0 .55rem; }
.sec span { width:6px; height:20px; border-radius:6px; background:#F4721E; display:inline-block; }
.sec-cap { color:#5E6878; font-size:.875rem; margin:-.3rem 0 .6rem; }

.kgrid { display:grid; gap:14px; margin-bottom:1rem; }
.kgrid.c4 { grid-template-columns:repeat(4, minmax(0, 1fr)); }
.kgrid.c3 { grid-template-columns:repeat(3, minmax(0, 1fr)); }
.kcard { background:#FFFFFF; border-radius:20px; padding:1rem 1.15rem; display:flex; flex-direction:column; min-height:138px;
  border-top:4px solid #4D6080; box-shadow:0 1px 2px rgba(30,42,59,.04), 0 10px 26px rgba(30,42,59,.06);
  transition:transform .15s ease, box-shadow .15s ease; }
.kcard:hover { transform:translateY(-2px); box-shadow:0 1px 2px rgba(30,42,59,.05), 0 16px 30px rgba(30,42,59,.10); }
.kcard.orange { border-top-color:#F4721E; }
.kcard.hl { border-top-color:transparent; color:#fff;
  background:radial-gradient(120% 120% at 100% 0%, #FFA266 0%, #F4721E 45%, #C4520F 100%); }
.kcard.blue { border-top-color:transparent; color:#fff; background:linear-gradient(150deg, #4D6080 0%, #3A4A66 100%); }
.k-label { font-size:.875rem; font-weight:600; color:#5E6878; }
.k-value { font-size:1.75rem; font-weight:700; letter-spacing:-.03em; color:#1E2A3B; margin:.35rem 0 .3rem; line-height:1.15; }
.k-sub { font-size:.875rem; color:#5E6878; margin-top:auto; line-height:1.4; }
.kcard.hl .k-label, .kcard.hl .k-sub, .kcard.blue .k-label, .kcard.blue .k-sub { color:rgba(255,255,255,.88); }
.kcard.hl .k-value, .kcard.blue .k-value { color:#fff; }

[class*="st-key-card-"] { background:#FFFFFF; border-radius:22px; padding:1.1rem 1.25rem 1.1rem !important;
  box-shadow:0 1px 2px rgba(30,42,59,.04), 0 10px 26px rgba(30,42,59,.06); }
.c-title { font-weight:700; font-size:1.02rem; color:#1E2A3B; }
.c-cap { color:#5E6878; font-size:.875rem; margin-top:.15rem; line-height:1.45; }
.dleg { margin-top:.2rem; }
.dl { display:grid; grid-template-columns:14px 1fr auto 52px; gap:.6rem; align-items:center; padding:.42rem 0;
  border-top:1px solid #EFE7DE; font-size:.875rem; }
.dl.off { opacity:.4; }
.dl i { width:12px; height:12px; border-radius:4px; display:inline-block; }
.dl-n { color:#1E2A3B; } .dl-v { color:#5E6878; text-align:right; } .dl-p { font-weight:700; text-align:right; color:#1E2A3B; }
.ins { display:flex; gap:.75rem; align-items:flex-start; padding:.6rem 0; border-top:1px solid #EFE7DE; font-size:.92rem; color:#1E2A3B; }
.ins:first-of-type { border-top:none; }
.ins-dot { width:10px; height:10px; border-radius:50%; background:#F4721E; margin-top:.4rem; flex:0 0 10px; }
[data-testid="stElementToolbar"] { display:none !important; }
.tip { background:#FFFFFF; border-radius:14px; padding:.6rem .9rem; font-size:.875rem; color:#5E6878; margin:.2rem 0 .4rem;
  border-left:4px solid #F4721E; }
.tip b { color:#1E2A3B; }
.st-key-chips { align-items:center; flex-wrap:wrap; margin:.2rem 0 .4rem; }
.chips-lab { font-size:.875rem; font-weight:700; color:#1E2A3B; margin-right:.2rem; }
.st-key-chips .stButton button { background:#FFFFFF; border:1.5px solid #F4721E; color:#B2470A; padding:.2rem .8rem;
  min-height:0; font-size:.875rem; }
.st-key-chips .stButton button:hover { background:#FDE3D0; }
.st-key-chips [class*="st-key-chip_clear"] button { background:#4D6080; border-color:#4D6080; color:#FFFFFF; }
.srow.wide { grid-template-columns:minmax(0, 1.5fr) minmax(80px, 1fr) 56px; }
.ogrid { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:14px; }
.ocard { background:#FFFFFF; border-radius:20px; padding:1rem 1.15rem; box-shadow:0 1px 2px rgba(30,42,59,.04), 0 10px 26px rgba(30,42,59,.06);
  display:flex; flex-direction:column; gap:.45rem; min-height:170px; }
.ocard .o-name { font-weight:700; font-size:.95rem; color:#1E2A3B; }
.ocard .o-val { font-size:1.6rem; font-weight:700; color:#1E2A3B; letter-spacing:-.02em; }
.ocard .o-val small { font-size:.875rem; font-weight:500; color:#5E6878; margin-left:.3rem; }
.otrack { height:10px; border-radius:6px; background:#F6F0EA; position:relative; overflow:hidden; }
.otrack div { position:absolute; left:0; top:0; bottom:0; border-radius:6px; background:#3B5C99; }
.o-cmp { display:flex; justify-content:space-between; font-size:.875rem; color:#5E6878; margin-top:auto; }
.o-cmp b { color:#1E2A3B; }
.st-key-tomodel a { background:#4D6080 !important; padding:.45rem 1.1rem !important; width:fit-content; max-width:100%; height:auto; }
.st-key-tomodel a p { color:#fff !important; font-weight:600; white-space:normal !important; }
@media (max-width: 1100px) {
  .kgrid.c4 { grid-template-columns:repeat(2, minmax(0, 1fr)); }
  .ogrid { grid-template-columns:repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  .kgrid.c4, .kgrid.c3, .ogrid { grid-template-columns:repeat(2, minmax(0, 1fr)); gap:10px; }
  [data-testid="stElementToolbar"] { display:none !important; }
.tip { background:#FFFFFF; border-radius:14px; padding:.6rem .9rem; font-size:.875rem; color:#5E6878; margin:.2rem 0 .4rem;
  border-left:4px solid #F4721E; }
.tip b { color:#1E2A3B; }
.st-key-chips { align-items:center; flex-wrap:wrap; margin:.2rem 0 .4rem; }
.chips-lab { font-size:.875rem; font-weight:700; color:#1E2A3B; margin-right:.2rem; }
.st-key-chips .stButton button { background:#FFFFFF; border:1.5px solid #F4721E; color:#B2470A; padding:.2rem .8rem;
  min-height:0; font-size:.875rem; }
.st-key-chips .stButton button:hover { background:#FDE3D0; }
.st-key-chips [class*="st-key-chip_clear"] button { background:#4D6080; border-color:#4D6080; color:#FFFFFF; }
.srow.wide { grid-template-columns:1fr 64px; }
  .kcard { min-height:128px; padding:.85rem .9rem; } .k-value { font-size:1.35rem; }
  .herob { padding:1.1rem 1.1rem; } .herob h1 { font-size:1.5rem; }
}
"""
