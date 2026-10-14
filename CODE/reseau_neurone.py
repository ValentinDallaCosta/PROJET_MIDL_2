from fonctions import *  # Importation des fonctions définies dans le fichier fonctions.py
from banque_image import *  # Importation des fonctions définies dans le fichier banque_image.py

#################################################
###### MISE EN PLACE DU RESEAU DE NEURONES ######
#################################################

input_size = 784 # Taille de l'entrée (nombre de pixels par image)
nb_couches = 3 # Nombre de couches dans le réseau de neurones
taille_couches = [input_size, 10, 10] # Taille de chaque couche (entrée, couche cachée, sortie)

#############################################################################
###### Première couche : Couche d'entrée (pixels de l'image de départ) ######
#############################################################################

# Exemple avec l'image 1 de la banque d'apprentissage
Couche_1 = recuperer_valeurs_pixels(1)  # Récupération des valeurs des pixels de l'image 1

######## Liste des paramètres à optimiser ########
# Poids des connections :
W_1 = np.random.randn(input_size, 10) * 0.01 
# Biais :
B_1 = np.random.uniform(-10, 10, (1, input_size))  # Biais entre -10 et 10


###########################################################
###### Deuxième couche : Couche cachée (10 neurones) ######
###########################################################

Couche_2 = np.zeros((1, 10))  # Initialisation de la couche cachée avec des zéros

# Apllication de la fonction d'activation sur la somme pondérée
Couche_2 = fonction_activation(somme_ponderee(W_1, Couche_1, B_1))
######## Liste des paramètres à optimiser ########
# Poids des connections :
W_2 = np.random.randn(10, 10) * 0.01 
# Biais entre -10 et 10
B_2 = np.random.uniform(-10, 10, (1, 10))

########################################################################################
####### Troisième couche : Couche de sortie (10 neurones pour les chiffres 0 à 9) ######
########################################################################################

Couche_3 = fonction_activation(somme_ponderee(W_2, Couche_2, B_2))  # Application de la fonction d'activation sur la somme pondérée


########### CALCUL DU GRANDIANT ET OPTIMISATION DES PARAMETRES ###########

