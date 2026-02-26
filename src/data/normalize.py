import pandas as pd
from sklearn.preprocessing import StandardScaler

print("Chargement des données pour la normalisation...")
# 1. Charger les données X
X_train = pd.read_csv('data/processed/X_train.csv')
X_test = pd.read_csv('data/processed/X_test.csv')

# --- LA CORRECTION EST ICI ---
# On exclut les colonnes de type "object" (texte/date) pour ne garder que les nombres
X_train = X_train.select_dtypes(exclude=['object'])
X_test = X_test.select_dtypes(exclude=['object'])
# -----------------------------

# 2. Initialiser le scaler
scaler = StandardScaler()

# 3. Normaliser les données
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# 4. Sauvegarder les nouveaux fichiers
X_train_scaled.to_csv('data/processed/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/processed/X_test_scaled.csv', index=False)

print("Normalisation terminée avec succès !")
