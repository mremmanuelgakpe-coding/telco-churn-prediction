# Données

Dataset **Telco Customer Churn** (Kaggle, `blastchar/telco-customer-churn`), jeu d'exemple d'IBM : 7 043 clients, 21 colonnes.

```bash
# CLI Kaggle (nécessite ~/.kaggle/kaggle.json, à ne jamais committer)
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw --unzip
# ou téléchargement manuel : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
# fichier attendu : data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Empreinte de contrôle (vérifiée au chargement) :
`sha256 = 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

Licence indiquée sur Kaggle : « Data files © Original Authors ». Le fichier n'est donc pas republié dans ce dépôt (`.gitignore`). Les fichiers de `reports/` ne contiennent que des agrégats.

| Variable | Description |
|---|---|
| customerID | Identifiant (non utilisé par le modèle) |
| gender, SeniorCitizen, Partner, Dependents | Profil du client |
| tenure | Ancienneté en mois |
| PhoneService, MultipleLines | Téléphonie |
| InternetService | DSL, Fiber optic ou No |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Options Internet |
| Contract | Month-to-month, One year, Two year |
| PaperlessBilling, PaymentMethod | Facturation |
| MonthlyCharges, TotalCharges | Facture mensuelle et total facturé ($) |
| **Churn** | **Cible** : Yes = résiliation dans le dernier mois |
