import os
from flask import Flask, request, render_template, jsonify
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

# --- GESTION DES CHEMINS RELATIFS ---
# On récupère le chemin absolu du dossier où se trouve app.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# On remonte d'un niveau (..) pour atteindre la racine du projet où se trouve mlruns
# Structure supposée : 
# ├── mlruns/
# └── app/
#     └── app.py
# root_dir = os.path.abspath(os.path.join(current_dir, "../src"))
mlruns_dir = os.path.join("mlruns")

# Configuration de l'URI de tracking avec formatage compatible Windows/Linux
mlflow.set_tracking_uri(f"file:///{mlruns_dir.replace(os.sep, '/')}")

def load_champion_model():
    exp_name = "Logistic_Regression"
    experiment = mlflow.get_experiment_by_name(exp_name)

    if experiment is None:
        # On liste les exps pour aider au debug si ça échoue
        available = [e.name for e in mlflow.search_experiments()]
        raise ValueError(f"Expérience '{exp_name}' introuvable dans {mlruns_dir}. "
                         f"Expériences détectées : {available}")
    
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id], 
        order_by=["metrics.f1_score DESC"], 
        max_results=1
    )
    
    if runs.empty:
        raise ValueError(f"Aucun run trouvé pour l'expérience {exp_name}")
        
    return mlflow.sklearn.load_model(f"runs:/{runs.iloc[0].run_id}/model")

# Chargement sécurisé au démarrage
model = None
try:
    model = load_champion_model()
    print(f"✅ Modèle chargé avec succès depuis : {mlruns_dir}")
except Exception as e:
    print(f"❌ Erreur de chargement : {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
        
    try:
        # On récupère les données du formulaire
        data = request.form.to_dict()
        
        # Conversion explicite des types (Flask reçoit tout en String)
        # Assure-toi que les noms correspondent exactement à ton X_train
        input_df = pd.DataFrame([{
            'income': float(data.get('income', 0)),
            'age': int(data.get('age', 0)),
            'years_employed': int(data.get('years_employed', 0)),
            'fico_score': int(data.get('fico_score', 0)),
            'loan_amount': float(data.get('loan_amount', 0)),
            'debt_to_income': float(data.get('debt_to_income', 0))
        }])
        
        prediction = model.predict(input_df)[0]
        # predict_proba peut aussi être utile pour afficher le % de risque
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # On utilise le port 5001 comme tu l'as demandé
    app.run(host='0.0.0.0', port=5001, debug=True)