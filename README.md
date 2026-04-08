# UI-Max — Interface Flask · Bankia Risk Engine

Branche dédiée à l'interface utilisateur Flask du projet MLOps.

## Structure

```
UI-Max/
├── app.py               ← Serveur Flask (routes / prédiction)
├── export_model.py      ← Export du meilleur modèle MLflow → app/best_model.pkl
├── requirements.txt
├── Procfile             ← Déploiement Gunicorn (AWS / Render)
└── templates/
    └── index.html       ← Interface sobre & pro (thème bancaire)
```

## Mise en route locale

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Exporter le meilleur modèle depuis MLflow
#    (nécessite d'avoir lancé src/model_engineering.py au préalable)
python export_model.py

# 3. Lancer l'app
python app.py
# → http://localhost:5000
```

## Déploiement AWS (Elastic Beanstalk)

```bash
# S'assurer que app/best_model.pkl est bien présent
eb init -p python-3.11 bankia-risk
eb create bankia-risk-env
eb deploy
```

## Features utilisées par le modèle

| Feature | Description |
|---|---|
| `credit_lines_outstanding` | Nombre de lignes de crédit actives |
| `loan_amt_outstanding` | Montant du prêt en cours ($) |
| `total_debt_outstanding` | Dette totale en cours ($) |
| `income` | Revenu annuel ($) |
| `years_employed` | Ancienneté professionnelle (années) |
| `fico_score` | Score FICO (300–850) |

**Auteur :** Max  
**Projet :** Bankia MLOps — DU Data Analytics, Sorbonne Paris 1
