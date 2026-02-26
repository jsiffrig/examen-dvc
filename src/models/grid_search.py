import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

print("1. Chargement des données d'entraînement normalisées...")
X_train = pd.read_csv('data/processed/X_train_scaled.csv')
# On utilise squeeze() car scikit-learn préfère une Série (1D) plutôt qu'un DataFrame pour la cible
y_train = pd.read_csv('data/processed/y_train.csv').squeeze() 

print("2. Lancement du GridSearch (cela peut prendre 1 à 2 minutes max)...")
# On choisit le RandomForest comme modèle de régression
model = RandomForestRegressor(random_state=42)

# Les paramètres que l'on veut tester (on reste simple)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10]
}

# Configuration et lancement de la recherche
grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"-> Meilleurs paramètres trouvés : {best_params}")

print("3. Sauvegarde des paramètres...")
# On sauvegarde le dictionnaire de paramètres dans un fichier .pkl
with open('models/best_params.pkl', 'wb') as f:
    pickle.dump(best_params, f)

print("GridSearch terminé ! Le fichier best_params.pkl est sauvegardé.")
