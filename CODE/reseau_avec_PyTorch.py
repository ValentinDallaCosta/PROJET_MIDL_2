import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gzip
import os
from banque_image import *

# ==========================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================
def charger_donnees_pytorch():
    # Charger les données d'entraînement et de test avec ratio de 90% pour l'entraînement et 10% pour le test
    X, Y = charger_donnees()

    # Convertir les données en tenseurs PyTorch
    X_train_tensor = torch.tensor(X[:9000], dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y[:9000], dtype=torch.long)  # Utiliser long pour les labels
    X_test_tensor = torch.tensor(X[9000:10000], dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y[9000:10000], dtype=torch.long)

    return X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor


# ==========================================
# 2. DÉFINITION DE L'ARCHITECTURE DU RÉSEAU
# ==========================================
class ReseauPyTorch(nn.Module):
    def __init__(self):
        super(ReseauPyTorch, self).__init__()
        # PyTorch gère automatiquement la création des poids et des biais
        self.couche_1 = nn.Linear(784, 16)  # nn.Linear = W*x + B
        self.couche_2 = nn.Linear(16, 16)
        self.couche_3 = nn.Linear(16, 10)
        
        # Fonction d'activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Propagation avant (forward pass) gérée automatiquement
        x = self.sigmoid(self.couche_1(x))
        x = self.sigmoid(self.couche_2(x))
        x = self.sigmoid(self.couche_3(x))
        return x

# ==========================================
# 3. ENTRAÎNEMENT DU RÉSEAU
# ==========================================
def entrainer_pytorch():
    X_data, Y_data, _, _ = charger_donnees_pytorch()
    
    # Instanciation du modèle
    modele = ReseauPyTorch()
    
    # Définition de la fonction de coût (MSE) et de l'optimiseur (SGD)
    criterium = nn.MSELoss()
    optimiseur = optim.SGD(modele.parameters(), lr=0.2)  # lr = taux d'apprentissage (alpha)
    
    nb_iterations = 20000
    
    print("Début de l'entraînement avec PyTorch...")
    for i in range(nb_iterations):
        index = i % 9000  # Pour parcourir les 9000 échantillons d'entraînement
        x_sample = X_data[index:index+1]
        y_true = Y_data[index]
        
        # Création du vecteur one-hot (PyTorch nécessite des dimensions correspondantes pour MSE)
        y_one_hot = torch.zeros((1, 10))
        y_one_hot[0, y_true] = 1.0

        # --- ÉTAPE 1 : Remise à zéro des gradients ---
        optimiseur.zero_grad()
        
        # --- ÉTAPE 2 : Passe avant (Forward) ---
        predictions = modele(x_sample)
        
        # --- ÉTAPE 3 : Calcul du coût ---
        cout = criterium(predictions, y_one_hot)
        
        # --- ÉTAPE 4 : Rétropropagation (Backward) ---
        # Cette simple ligne remplace ta fonction 'calculer_gradiant' entière !
        cout.backward()
        
        # --- ÉTAPE 5 : Mise à jour des poids et biais ---
        # Cette ligne remplace ta fonction 'mise_a_jour_parametres' entière !
        optimiseur.step()
        
        if (i + 1) % 5000 == 0:
            print(f"Itération {i + 1}/{nb_iterations} terminée | Coût : {cout.item():.4f}")

    print("Entraînement terminé !")
    return modele


def tester_modele(modele, X_test, Y_test):
    modele.eval()  # Mettre le modèle en mode évaluation
    with torch.no_grad():  # Pas de calcul de gradients nécessaire pour l'évaluation
        predictions = modele(X_test)
        _, predicted_labels = torch.max(predictions, 1)  # Obtenir les indices des classes prédites
        accuracy = (predicted_labels == Y_test).float().mean()  # Calculer la précision
    return accuracy.item()


x_train, y_train, X_test, Y_test = charger_donnees_pytorch()
modele = entrainer_pytorch()
accuracy = tester_modele(modele, X_test, Y_test)
print(f"Précision sur les données de test : {accuracy:.4f}")