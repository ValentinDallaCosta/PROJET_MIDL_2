from fonctions import * 
from banque_image import *  
import numpy as np

#################################################
###### MISE EN PLACE DU RESEAU DE NEURONES ######
#################################################

charger_banque_apprentissage()
X_data, Y_data = charger_donnees()

input_size = 784
taille_couches = [input_size, 16, 10]
alpha = 0.3  # Un taux d'apprentissage 

######## Liste des paramètres à optimiser ########
# Initialisé aléatoirement :
# Poids des connections :
W = initialiser_poids(taille_couches)  # Initialisation de Xavier des poids

# Biais :
B = initialiser_biais(taille_couches)  # Initialisation des biais à zéro

# Separation des données d'entraînement et de test
# 90% des données pour l'entraînement et 10% pour le test
X_train = X_data[:9000]
Y_train = Y_data[:9000]
X_test = X_data[9000:10000]
Y_test = Y_data[9000:10000]

"""
# Entraînement du réseau de neurones sur l'ensemble des données d'entraînement
W, B = entrainer_reseau_neurone(X_train, Y_train, W, B, alpha, taille_couches, 100000)

# Test global sur l'ensemble des données de test
print("\nTest global sur l'ensemble des données de test :")
taux_reussite, erreur_moyenne = test_global_reseau_neurone(X_test, Y_test, W, B)
print(f"Taux de réussite : {taux_reussite} %")
print(f"Erreur moyenne : {erreur_moyenne:.4f}")"""

#####################################################################################
### Boucle d'entrainement afin d'optimiser les paramètres du réseau de neurones : ###
######################################################################################

# Variation du taux d'apprentissage alpha pour observer son impact sur la convergence du réseau de neurones :

# Il est préférable de l'avoir entre 0.1 et 0.5 pour un apprentissage efficace, mais pas trop rapide pour éviter les oscillations.


# Variation du nombre d'itérations pour observer son impact sur la convergence du réseau de neurones : 
for k in range (13):
    nb_iterations = 20000 * (k+1)  # Augmentation du nombre d'itérations
    #arret de l'affichage pour ne pas saturer la console
    W = initialiser_poids(taille_couches)  # Initialisation de Xavier des poids
    B = initialiser_biais(taille_couches)  # Initialisation des biais à zéro
    W, B = entrainer_reseau_neurone(X_train, Y_train, W, B, alpha, taille_couches, nb_iterations)

    # Test global sur l'ensemble des données de test après l'entraînement
    taux_reussite, erreur_moyenne = test_global_reseau_neurone(X_test, Y_test, W, B)
    print(f"Taux de réussite après {nb_iterations} images d'entraînements : {taux_reussite} %")
    print(f"Erreur moyenne après {nb_iterations} images d'entraînements : {erreur_moyenne:.4f}")
    print("--------------------------------------------------")
