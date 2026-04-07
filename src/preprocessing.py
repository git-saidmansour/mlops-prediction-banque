import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Dossiers de sortie
os.makedirs("data/processed", exist_ok=True)
os.makedirs("app", exist_ok=True)

print("🚀 Chargement de Loan_Data.csv...")
df = pd.read_csv("data/Loan_Data.csv")

# 2. Nettoyage
df = df.dropna()

# 3. Préparation (X, y)
# Note : Dans votre CSV, la cible est la dernière colonne
target_col = df.columns[-1]
X = df.drop(columns=[target_col])
y = df[target_col]

# 4. Split et Scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Sauvegarde
pd.DataFrame(X_train_scaled).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test_scaled).to_csv("data/processed/X_test.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
joblib.dump(scaler, "app/preprocessor.pkl")

print("✅ Données prêtes dans data/processed/ et dossier src/ activé !")