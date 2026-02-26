import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

print("1. Chargement des données et des paramètres...")
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
y_train = pd.read_csv('data/processed/y_train.csv').squeeze()

# On charge le dictionnaire de paramètres qu'on a créé à l'étape précédente
with open('models/best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

print(f"2. Entraînement du modèle avec : {best_params}...")
# Les deux étoiles ** permettent de déballer le dictionnaire directement dans le modèle
model = RandomForestRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)

print("3. Sauvegarde du modèle entraîné...")
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Entraînement terminé ! Le modèle est sauvegardé sous models/model.pkl")
