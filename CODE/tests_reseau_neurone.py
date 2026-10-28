from fonctions import main_reseau
input_size = 784 # Nb de pixels sur l'image de départ
output_size = 10 # Correspond au 10 chiffer de prédiction en sortie

####################################################################
### Début des tests sur les paramètres du réseaux de neuronens : ###
####################################################################

# On fera varier : - Le nombre de couche cachée 
#                  - La taille des couches cachées
#                  - Le taux d'apprentissage alpha
#                  - Le nombre d'image étudiée lors de l'apprentissage


# Tests 1 :
print("--------------------- TEST 1 ---------------------")
tailles_couches = [input_size, 16, output_size] # 3 couches de tailles 784, 16 et 10
alpha = 0.3 # Taux d'apprentissage
nb_image_train = 100
print(f"Couches = {tailles_couches} ------ alpha = {alpha} ------ {nb_image_train} images d'entraînement")
main_reseau(tailles_couches, alpha, nb_image_train)

# Tests 1 :
print("--------------------- TEST 2 ---------------------")
tailles_couches = [input_size, 16, output_size] # 3 couches de tailles 784, 16 et 10
alpha = 0.3 # Taux d'apprentissage
nb_image_train = 10000000
print(f"Couches = {tailles_couches} ------ alpha = {alpha} ------ {nb_image_train} images d'entraînement")
main_reseau(tailles_couches, alpha, nb_image_train)
