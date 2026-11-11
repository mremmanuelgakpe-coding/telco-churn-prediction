"""Figures statiques (PNG) du rapport. Palette orange / noir / gris, comme la démo."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from .explain import LABELS_FR  # noqa: E402

ORANGE, BLUE, INK, INK2, GRID, MUTED, SURFACE = "#E85A0C", "#2F6CA8", "#171717", "#6B6A66", "#E6E5E2", "#C9C7C2", "#FCFCFB"
SERIES = [ORANGE, INK, BLUE, "#8A8782"]  # 4 courbes au maximum, ordre fixe
LABEL = {
    "logistic_regression": "Régression logistique", "svm": "SVM (RBF)", "knn": "KNN", "naive_bayes": "Naive Bayes",
    "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting", "xgboost": "XGBoost",
    "lightgbm": "LightGBM", "catboost": "CatBoost",
}


def _style(ax, title):
    ax.figure.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def comparison_bar(comp, selected, path):
    df = comp.sort_values("cv_roc_auc")
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    colors = [ORANGE if m == selected else MUTED for m in df["model"]]
    ax.barh([LABEL[m] for m in df["model"]], df["cv_roc_auc"], xerr=df["cv_roc_auc_std"], color=colors,
            height=0.62, error_kw={"ecolor": INK2, "lw": 1})
    for i, (v, sd) in enumerate(zip(df["cv_roc_auc"], df["cv_roc_auc_std"])):
        ax.text(v + sd + 0.003, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=9, color=INK)
    ax.set_xlim(0.70, 0.88)
    ax.set_xlabel("ROC AUC (validation croisée 5 plis)")
    _style(ax, "Comparaison des 9 classifieurs (orange : modèle retenu)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def roc_pr(y, probas: dict, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for color, (name, p) in zip(SERIES, probas.items()):
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, lw=2, color=color, label=LABEL[name])
        pr, rc, _ = precision_recall_curve(y, p)
        a2.plot(rc, pr, lw=2, color=color, label=LABEL[name])
    a1.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2)
    a2.axhline(np.mean(y), ls="--", lw=1, color=INK2)
    _style(a1, "Courbes ROC (jeu de test)")
    _style(a2, "Précision-Rappel (jeu de test)")
    a1.set_xlabel("Taux de faux positifs")
    a1.set_ylabel("Taux de vrais positifs")
    a2.set_xlabel("Rappel (churners détectés)")
    a2.set_ylabel("Précision")
    a2.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    _save(fig, path)


def profit_curve(curve, best, path):
    t = [c[0] for c in curve]
    v = [c[1] for c in curve]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(t, v, lw=2, color=ORANGE)
    ax.axvline(best, ls="--", lw=1, color=INK2)
    ax.axhline(0, lw=1, color=GRID)
    ax.text(best + 0.01, max(v) * 0.92, f"seuil retenu = {best:.2f}".replace(".", ","), color=INK, fontsize=9)
    ax.set_xlabel("Seuil de ciblage (probabilité de churn)")
    ax.set_ylabel("Valeur nette attendue ($, clients d'entraînement)")
    _style(ax, "Valeur de la campagne de rétention selon le seuil (hors test)")
    _save(fig, path)


def calibration(y, p, path):
    frac, mean = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=INK2, label="Calibration parfaite")
    ax.plot(mean, frac, marker="o", ms=6, lw=2, color=ORANGE, label="Modèle retenu")
    ax.set_xlabel("Probabilité de churn prédite")
    ax.set_ylabel("Taux de churn observé")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _style(ax, "Calibration (déciles de score)")
    _save(fig, path)


def shap_global(names, mean_abs, path, top=12):
    order = np.argsort(mean_abs)[-top:]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.barh([LABELS_FR.get(names[i], names[i]) for i in order], [mean_abs[i] for i in order], color=ORANGE,
            height=0.62)
    ax.set_xlabel("|SHAP| moyen (log-odds)")
    _style(ax, "Ce qui pèse le plus sur le risque de churn (SHAP, test)")
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def shap_example(values: dict, contribs: dict, base_p, p, path):
    items = sorted(contribs.items(), key=lambda kv: abs(kv[1]))[-10:]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.1f}".replace(".", ",") if not v.is_integer() else f"{v:.0f}"
        return str(v)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.barh([f"{LABELS_FR.get(k, k)} = {fmt(values[k])}" for k, _ in items], [v for _, v in items],
            color=[ORANGE if v > 0 else BLUE for _, v in items], height=0.62)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Contribution SHAP (log-odds) — à droite : augmente le risque")
    _style(ax, f"Exemple de client à risque : {p:.0%} (score de base {base_p:.0%})".replace("%", " %"))
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def churn_by(df, col, target, path, title, order=None):
    g = df.groupby(col, observed=True)[target].mean()
    if order:
        g = g.reindex(order)
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    ax.bar(g.index.astype(str), g.values, color=ORANGE, width=0.6)
    for i, v in enumerate(g.values):
        ax.text(i, v + 0.01, f"{v:.0%}".replace("%", " %"), ha="center", fontsize=9, color=INK)
    ax.set_ylabel("Taux de churn")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}".replace("%", " %")))
    _style(ax, title)
    ax.grid(axis="x", visible=False)
    _save(fig, path)
