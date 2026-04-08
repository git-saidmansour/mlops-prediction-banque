"""
export_model.py
---------------
Récupère le meilleur modèle (Random Forest) depuis MLflow et le sauvegarde
en app/best_model.pkl pour que l'interface Flask puisse le charger.

Usage :
    python export_model.py
    (depuis la racine du projet, après avoir lancé model_engineering.py)
"""

import os
import mlflow
import joblib

# ── Config ──────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "sqlite:///src/mlflow.db"   # chemin relatif à la racine
EXPERIMENT_NAME     = "Random_Forest"              # meilleur modèle attendu
OUTPUT_PATH         = os.path.join("app", "best_model.pkl")
# ─────────────────────────────────────────────────────────────────────────────

def export_best_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise ValueError(
            f"Experiment '{EXPERIMENT_NAME}' introuvable. "
            "Lance d'abord src/model_engineering.py"
        )

    # Récupère le run avec le meilleur F1
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("Aucun run trouvé dans l'experiment.")

    best_run = runs[0]
    run_id   = best_run.info.run_id
    f1       = best_run.data.metrics.get("f1_score", "N/A")
    print(f"✅ Meilleur run : {run_id}  |  F1 = {f1:.4f}")

    # Charge le pipeline sklearn depuis MLflow
    model_uri = f"runs:/{run_id}/model"
    pipeline  = mlflow.sklearn.load_model(model_uri)

    # Sauvegarde en pkl
    os.makedirs("app", exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"✅ Modèle sauvegardé → {OUTPUT_PATH}")


if __name__ == "__main__":
    export_best_model()
