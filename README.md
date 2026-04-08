# mlops-prediction-banque

Ce projet implémente la partie "Model Engineering" pour la prédiction du défaut de paiement (`default`) à partir des données de prêt.

## Contenu

- `src/preprocessing.py` : script de préparation des données.
- `src/model_engineering.py` : script d'entraînement et de tracking MLflow pour trois modèles de classification.
- `data/Loan_Data.csv` : jeu de données principal.
- `requirements.txt` : dépendances Python.

## Modèles testés

- `LogisticRegression`
- `DecisionTree`
- `RandomForest`

## Utilisation

1. Installer les dépendances :

```bash
pip install -r requirements.txt
```

2. Lancer l'entraînement et le tracking MLflow :

```bash
python src/model_engineering.py
```

3. Ouvrir l'interface MLflow :

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

4. Aller sur `http://127.0.0.1:5000` pour visualiser les experiments et les runs.

## Organisation MLflow

- Un experiment MLflow par modèle.
- Une run MLflow par exécution du script / itération de modèle.
- Les métriques trackées : `accuracy`, `precision`, `recall`, `f1_score`.
- Le modèle et la confusion matrix sont enregistrés comme artifacts.
