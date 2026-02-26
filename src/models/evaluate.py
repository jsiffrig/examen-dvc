import pandas as pd
import pickle
import json
from sklearn.metrics import mean_squared_error, r2_score

print("1. Chargement du modèle et des données de test...")
# On charge le modèle entraîné
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# On charge les données de test (X doit être normalisé !)
X_test = pd.read_csv('data/processed/X_test_scaled.csv')
y_test = pd.read_csv('data/processed/y_test.csv').squeeze()

print("2. Réalisation des prédictions...")
predictions = model.predict(X_test)

# Sauvegarde des prédictions dans data/
df_preds = pd.DataFrame({'y_true': y_test, 'y_pred': predictions})
df_preds.to_csv('data/prediction.csv', index=False)

print("3. Calcul et sauvegarde des métriques...")
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Préparation du dictionnaire pour le fichier JSON
scores = {
    'mse': mse,
    'r2': r2
}

# Sauvegarde dans metrics/scores.json
with open('metrics/scores.json', 'w') as f:
    json.dump(scores, f, indent=4)

print(f"Évaluation terminée ! MSE: {mse:.4f} | R2: {r2:.4f}")
print("Fichiers 'data/prediction.csv' et 'metrics/scores.json' générés avec succès.")
