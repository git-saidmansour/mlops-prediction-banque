import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

DATA_PATH = os.path.join("data", "Loan_Data.csv")

# Dictionnaire des modèles
MODELS = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(max_depth=5), # Itération 1
    "Random_Forest": RandomForestClassifier(n_estimators=100),
}

def train_and_track():
    # 1. Chargement et préparation (ton code actuel)
    df = pd.read_csv(DATA_PATH).dropna()
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])
    
    X = df.drop(columns=["default"])
    y = df["default"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Boucle d'entraînement
    for model_name, model in MODELS.items():
        # CONSIGNE : Un modèle -> un experiment
        mlflow.set_experiment(model_name)
        
        # CONSIGNE : Itération -> un run
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        with mlflow.start_run(run_name=run_name):
            # Création du Pipeline (Clean & Scalable)
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model),
            ])
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # Métriques
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred)
            }

            # Logging
            mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            
            # Sauvegarde du modèle (Artéfact)
            mlflow.sklearn.log_model(pipeline, "model")

            print(f"✅ {model_name} enregistré dans MLflow (F1: {metrics['f1_score']:.4f})")

if __name__ == "__main__":
    train_and_track()