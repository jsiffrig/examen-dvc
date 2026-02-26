import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. Créer le dossier de destination au cas où
os.makedirs('data/processed', exist_ok=True)

# 2. Charger les données brutes
print("Chargement des données brutes...")
df = pd.read_csv('data/raw/raw.csv')

# 3. Séparer les features (X) de la target (y)
# La consigne précise que la cible est 'silica_concentrate'
X = df.drop(columns=['silica_concentrate'])
y = df['silica_concentrate']

# 4. Découper en Train (80%) et Test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Sauvegarder les 4 nouveaux fichiers
X_train.to_csv('data/processed/X_train.csv', index=False)
X_test.to_csv('data/processed/X_test.csv', index=False)
y_train.to_csv('data/processed/y_train.csv', index=False)
y_test.to_csv('data/processed/y_test.csv', index=False)

print("Découpage terminé avec succès !")
