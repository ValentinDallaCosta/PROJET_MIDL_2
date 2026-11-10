from fonctions import main_reseau
import numpy as np

input_size = 784  # Nombre de pixels de l'image
output_size = 10  # Nombre de classes à prédire
alpha = 0.3

####################################################################
### Début des tests sur les paramètres du réseau de neurones :    ###
####################################################################

# On fera varier : - le nombre d'itérations (nombre d'images d'apprentissage)
#                  - le nombre de couhes
#                  - le nombre de neurones par couche

# Les tests 1 à 3 gardent la même architecture et font varier le nombre
# d'itérations. Les tests 3 à 5 gardent le même nombre d'itérations et
# comparent la largeur puis la profondeur du réseau.

# Test 1 :
print("--------------------- TEST 1 ---------------------")
tailles_couches = [input_size, 16, output_size]
nb_image_train = 100
print(f"Couches = {tailles_couches} ------ alpha = {alpha} ------ {nb_image_train} itérations")
main_reseau(tailles_couches, alpha, nb_image_train)

# Test 2 :
print("--------------------- TEST 2 ---------------------")
tailles_couches = [input_size, 16, output_size]
nb_image_train = 1000
print(f"Couches = {tailles_couches} ------ alpha = {alpha} ------ {nb_image_train} itérations")
main_reseau(tailles_couches, alpha, nb_image_train)

# Test 3 :
print("--------------------- TEST 3 ---------------------")
tailles_couches = [input_size, 16, output_size]
nb_image_train = 9000
print(f"Couches = {tailles_couches} ------ alpha = {alpha} ------ {nb_image_train} itérations")
main_reseau(tailles_couches, alpha, nb_image_train)

# Test 4 :
print("--------------------- TEST 4 ---------------------")
tailles_couches = [input_size, 16, output_size]
nb_image_train = 15000
print(f"Couches = {tailles_couches} ------ alpha = {alpha} ------ {nb_image_train} itérations")
main_reseau(tailles_couches, alpha, nb_image_train)

# Test 5 :
print("--------------------- TEST 5 ---------------------")
tailles_couches = [input_size, 64, output_size]
nb_image_train = 9000
print(f"Couches = {tailles_couches} ------ alpha = {alpha} ------ {nb_image_train} itérations")
main_reseau(tailles_couches, alpha, nb_image_train)

# Test 6 :
print("--------------------- TEST 6 ---------------------")
tailles_couches = [input_size, 16, 16, output_size]
nb_image_train = 9000
print(f"Couches = {tailles_couches} ------ alpha = {alpha} ------ {nb_image_train} itérations")
main_reseau(tailles_couches, alpha, nb_image_train)
