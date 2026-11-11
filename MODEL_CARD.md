# Fiche modèle : Telco Churn v1.0.0

| Élément | Détail |
|---|---|
| Tâche | Classification binaire : probabilité de résiliation dans le mois |
| Modèle | Régression logistique (C = 0,5) dans un pipeline scikit-learn (standardisation, log de TotalCharges, one-hot) |
| Variables | 19 (profil, services, contrat, facturation) ; identifiant client exclu |
| Données | Telco Customer Churn (IBM / Kaggle), 7 043 clients, 26,5 % de churn |
| Découpage | 80 % entraînement (validation croisée stratifiée 5 plis) / 20 % test, graine 42 |
| Seuil de ciblage | 0,27, maximisant la valeur d'une campagne simulée (offre = 1 mois, succès 30 %, 12 mois conservés), choisi hors test |
| Traçabilité | Empreintes SHA-256 des données et du modèle dans `models/model.meta.json` |

## Performances (test, 1 409 clients)

| ROC AUC | PR-AUC | Brier | Lift top 20 % | Départs captés dans le top 20 % |
|---|---|---|---|---|
| 0,848 | 0,664 | 0,135 | 2,56 | 51 % |

## Usage prévu
Démonstration de portfolio : prioriser des actions de rétention et expliquer chaque score. **Pas** de décision automatique visant un client réel sans validation sur les données de l'opérateur, ni sans mesure de l'efficacité de l'offre par un test A/B.

## Limites et risques
- Données d'un seul mois, sans horodatage : pas de prédiction du moment du départ, stabilité dans le temps non évaluée.
- Le genre et le statut senior figurent parmi les variables : leur usage pour cibler une offre commerciale doit être justifié (équité, RGPD) ; le genre pèse très peu dans le modèle.
- Ancienneté et total facturé sont corrélés : les contributions SHAP se partagent leur effet.
- Les hypothèses économiques sont illustratives et réglables.

## Réentraînement
`make train` : régénère modèle, métadonnées, métriques et figures de façon reproductible.
