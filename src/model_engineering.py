import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import mlflow.sklearn

DATA_PATH = os.path.join("data", "Loan_Data.csv")
MLFLOW_TRACKING_DIR = os.path.join(os.getcwd(), "mlruns")

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver="liblinear"),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
}


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_data(df: pd.DataFrame):
    df = df.dropna()
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    target_col = "default"
    if target_col not in df.columns:
        raise ValueError(f"La colonne cible '{target_col}' est introuvable dans le dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def evaluate(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }


def save_confusion_matrix(y_true, y_pred, path: str):
    cm = confusion_matrix(y_true, y_pred)
    with open(path, "w", encoding="utf-8") as f:
        f.write("confusion_matrix\n")
        f.write("tn, fp, fn, tp\n")
        f.write(", ".join(str(x) for x in cm.ravel()) + "\n")


def train_and_track():
    os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_DIR)

    df = load_data(DATA_PATH)
    X, y = prepare_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    for model_name, model in MODELS.items():
        mlflow.set_experiment(model_name)
        with mlflow.start_run(run_name=f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"):
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("model", model),
            ])
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)
            metrics = evaluate(y_test, y_pred)

            for param_name, param_value in model.get_params().items():
                mlflow.log_param(param_name, param_value)

            mlflow.log_params({"train_size": len(X_train), "test_size": len(X_test)})
            mlflow.log_metrics(metrics)

            mlflow.sklearn.log_model(pipeline, artifact_path="model")

            os.makedirs("mlruns_artifacts", exist_ok=True)
            cm_path = os.path.join("mlruns_artifacts", f"confusion_matrix_{model_name}.csv")
            save_confusion_matrix(y_test, y_pred, cm_path)
            mlflow.log_artifact(cm_path)

            print(f"✅ Experiment '{model_name}' tracée : accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}")


def main():
    train_and_track()


if __name__ == "__main__":
    main()
