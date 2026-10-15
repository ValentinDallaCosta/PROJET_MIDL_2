from fonctions import *  # Importation des fonctions définies dans le fichier fonctions.py
from banque_image import *  # Importation des fonctions définies dans le fichier banque_image.py

#################################################
###### MISE EN PLACE DU RESEAU DE NEURONES ######
#################################################

charger_banque_apprentissage()
# Chargement unique en mémoire RAM
X_data, Y_data = charger_donnees()


input_size = 784 # Taille de l'entrée (nombre de pixels par image)
nb_couches = 4 # Nombre de couches dans le réseau de neurones
taille_couches = [input_size, 16, 16,10] # Taille de chaque couche (entrée, couche cachée, sortie)
alpha = 0.1 # Taux d'apprentissage pour la mise à jour des poids et biais


######## Liste des paramètres à optimiser ########
# Initialisé aléatoirement :
# Poids des connections :
W_1 = np.random.randn(input_size, taille_couches[1]) * 0.01 
W_2 = np.random.randn(taille_couches[1], taille_couches[2]) * 0.01
W_3 = np.random.randn(taille_couches[2], taille_couches[3]) * 0.01

# Biais :
B_1 = np.zeros((1, taille_couches[1]))
B_2 = np.zeros((1, taille_couches[2]))
B_3 = np.zeros((1, taille_couches[3]))

for I in range(1000):  # Boucle d'entraînement
    index = I % 10000  # Assurez-vous que l'index reste dans les limites de la banque d'images
    print(f"--- Itération {I+1} ---")

    ###### Première couche : Couche d'entrée (pixels de l'image de départ) ######
    Couche_1 = X_data[index:index+1]  # Forme (1, 784)
    vrai_chiffre = Y_data[index]      # Entier du chiffre
    
    Couche_1 = recuperer_valeurs_pixels(index).reshape(1, input_size) # Matrice de taille (1, 784)

    ###### Deuxième couche : Couche cachée (16 neurones) ######
    Z_2 = somme_ponderee(Couche_1, W_1, B_1)  # Somme pondérée pour la couche cachée    
    Couche_2 = fonction_activation(Z_2)  # Application de la fonction d'activation sur la somme pondérée
    
    ####### Troisième couche : Couche cachée (16 neurones) ######
    Z_3 = somme_ponderee(Couche_2, W_2, B_2)  # Somme pondérée pour la couche de sortie
    Couche_3 = fonction_activation(Z_3)  # Application de la fonction d'activation sur la somme pondérée

    ####### Quatrième couche : Couche de sortie (10 neurones pour les chiffres 0-9) ######
    Z_4 = somme_ponderee(Couche_3, W_3, B_3)  # Somme pondérée pour la couche de sortie
    Couche_4 = fonction_activation(Z_4)  # Application de la fonction d'activation sur la somme pondérée


    # Calcul de la fonction de coût
    etiquette_vecteur = one_hot_encode(vrai_chiffre)  # Vecteur one-hot pour l'étiquette réelle
    erreur = cout_mse(Couche_4, etiquette_vecteur)  # Calcul de l'erreur entre la sortie du réseau et le vecteur d'étiquette


    ########### CALCUL DU GRADIENT ET OPTIMISATION DES PARAMETRES ###########
    # Le gradient n'est pas calculé par rapport à l'entrée X, mais par rapport aux paramètres entraînables du modèle (poids et biais). Le gradient est calculé pour chaque paramètre du modèle, et non pour chaque exemple d'entrée. Le gradient est ensuite utilisé pour mettre à jour les paramètres du modèle afin de minimiser la fonction de coût.:

    # --- Gradients ---
    # Erreur en sortie
    delta_3 = (Couche_4 - etiquette_vecteur) * sigmoid_derivative(Z_4)
    dW_3 = np.dot(Couche_3.T, delta_3)
    dB_3 = np.sum(delta_3, axis=0, keepdims=True)

    # Erreur sur la couche cachée
    delta_2 = np.dot(delta_3, W_3.T) * sigmoid_derivative(Z_3)
    dW_2 = np.dot(Couche_2.T, delta_2)
    dB_2 = np.sum(delta_2, axis=0, keepdims=True)

    # Erreur sur la couche cachée
    delta_1 = np.dot(delta_2, W_2.T) * sigmoid_derivative(Z_2)
    dW_1 = np.dot(Couche_1.T, delta_1)
    dB_1 = np.sum(delta_1, axis=0, keepdims=True)

    # --- 4. MISE À JOUR DES PARAMÈTRES ---
    W_3 -= alpha * dW_3
    B_3 -= alpha * dB_3
    W_2 -= alpha * dW_2
    B_2 -= alpha * dB_2
    W_1 -= alpha * dW_1
    B_1 -= alpha * dB_1
    #print(f"sortie du réseau : {Couche_4}")

afficher_image_banque(999)  # Affiche l'image correspondant à l'index actuel
print(f"Coût calculé : {erreur:.4f}")
print(f"Sortie du réseau : {Couche_4}")