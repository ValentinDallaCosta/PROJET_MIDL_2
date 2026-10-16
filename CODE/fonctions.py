import numpy as np

#############################
### Fonction d'activation ###
#############################
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def fonction_activation(matrix):
    """
    Applique la fonction sigmoïde à chaque élément de la matrice.
    matrix : matrice d'entrée
    """
    return sigmoid(matrix)  # Application de la fonction sigmoïde à chaque élément de la matrice


#################################
####  Fonction Somme pondérée ###
#################################
def somme_ponderee(A,W,B):
    """
    Calcule la somme pondérée des entrées.
    W : matrice des poids (taille : input_size x 10)
    A : matrice des activations (taille : nombre d'exemples x input_size)
    B : vecteur des biais (taille : 1 x 10)
    """
    return np.dot(A, W) + B  # Somme pondérée pour chaque exemple

########################
### Fonction de coût ###
########################
def cout_mse(X, Y):
    """
    Fonction de coût pour le réseau de neurones.
    X : matrice d'entrée 
    Y : vecteur des étiquettes (taille : nombre d'exemples)
    """
    cout = 1/2 * np.sum((X-Y)**2)
    return cout

################################################
### Fonction de vectorisation de l'étiquette ###
################################################
def one_hot(y):
    """
    Transforme un entier en vecteur one-hot.
    y : entier représentant l'étiquette (0 à 9)
    Retourne un vecteur de taille 10 avec un 1 à la position de l'étiquette et des 0 ailleurs.
    """
    vecteur_etiquette = np.zeros(10)
    vecteur_etiquette[y] = 1
    return vecteur_etiquette


#####################################################
### Fonctions de création des paramètres initiaux ###
#####################################################
def initialiser_poids(taille_couches):
    """
    Initialise les poids du réseau de neurones.
    Utilise la méthode de Xavier pour l'initialisation.
    taille_couches : liste contenant le nombre de neurones dans chaque couche
    Retourne une liste de matrices de poids pour chaque couche.
    """
    liste_W = []
    for i in range(len(taille_couches) - 1):
        W = np.random.randn(taille_couches[i], taille_couches[i + 1]) * np.sqrt(2.0 / taille_couches[i])  # Initialisation de Xavier
        liste_W.append(W)
    return liste_W

def initialiser_biais(taille_couches):
    """
    Initialise les biais du réseau de neurones.
    taille_couches : liste contenant le nombre de neurones dans chaque couche
    Retourne une liste de vecteurs de biais pour chaque couche.
    """
    liste_B = []
    for i in range(1, len(taille_couches)):
        B = np.zeros((1, taille_couches[i]))  # Initialisation des biais à zéro
        liste_B.append(B)
    return liste_B


######################################################
### Fonctions d'entrainement du réseau de neurones ###
######################################################
def entrainer_reseau_neurone(X_data, Y_data, liste_W, liste_B, alpha, taille_couches, nb_iterations):
    """
    Fonction principale du réseau de neurones.
    X_data : matrice d'entrée (taille : nombre d'exemples x input_size)
    Y_data : vecteur des étiquettes (taille : nombre d'exemples)
    liste_W : liste des matrices des poids pour chaque couche
    liste_B : liste des vecteurs des biais pour chaque couche
    alpha : taux d'apprentissage
    """
    nb_couches = len(taille_couches)
    nb_images = X_data.shape[0]  # Nombre d'images dans les données d'entraînement
    
    for iteration in range(nb_iterations):
        index = iteration % nb_images  # On s'assure que l'index reste dans les limites de la banque d'images

        X_sample = X_data[index:index+1]  # (1, 784)
        etiquette = Y_data[index]  # Entier du chiffre associé à l'image


        liste_Z = []  # Liste pour stocker les sommes pondérées de chaque couche
        couche_1 = X_sample  # Couche d'entrée
        couches = [couche_1]  # Liste pour stocker les activations de chaque couche

        for i in range(1, nb_couches):
            Z_i = somme_ponderee(couches[i-1], liste_W[i-1], liste_B[i-1])  # Somme pondérée pour la couche cachée
            liste_Z.append(Z_i)  # Stocke la somme pondérée  
            couche_i = fonction_activation(Z_i)  # Application de la fonction d'activation pour calculer la couche suivante
            couches.append(couche_i)

        delta = calculer_gradiant(liste_W, liste_B, couches, liste_Z, one_hot(etiquette))
        liste_W, liste_B = mise_a_jour_parametres(liste_W, liste_B, couches, delta, alpha)

        #if (iteration+ 1) % 5000 == 0:
                #print(f"Itération {iteration + 1}/{nb_iterations} terminée")


    #print("\nEntraînement terminé !")
    return liste_W, liste_B  # Retourne les poids et biais mis à jour

def calculer_gradiant(liste_poids, liste_biais, liste_couches, liste_Z, etiquette_vecteur):
    """
    Calcule le gradient pour chaque couche du réseau de neurones.
    liste_poids : liste des matrices des poids pour chaque couche
    liste_biais : liste des vecteurs des biais pour chaque couche
    liste_couches : liste des activations de chaque couche
    liste_Z : liste des sommes pondérées de chaque couche
    etiquette_vecteur : vecteur one-hot représentant l'étiquette réelle
    """

    nb_couches = len(liste_couches)
    delta = [None] * nb_couches  # Liste pour stocker les deltas pour chaque couche

    delta[-1] = (liste_couches[-1] - etiquette_vecteur) * sigmoid_derivative(liste_Z[-1])  # Erreur en sortie

    for i in range(nb_couches - 2, 0, -1):  # Boucle pour les couches cachées
        delta[i] = np.dot(delta[i + 1], liste_poids[i].T) * sigmoid_derivative(liste_Z[i - 1])  # Erreur pour la couche cachée

    return delta  # Retourne la liste des deltas pour chaque couche

def mise_a_jour_parametres(liste_poids, liste_biais, liste_couches, delta, alpha):
    """
    Met à jour les paramètres du réseau de neurones.
    liste_poids : liste des matrices des poids pour chaque couche
    liste_biais : liste des vecteurs des biais pour chaque couche
    liste_couches : liste des activations de chaque couche
    delta : liste des deltas pour chaque couche
    alpha : taux d'apprentissage
    Retourne les poids et biais mis à jour.
    """
    nb_couches = len(liste_couches)
    for i in range(1, nb_couches):
        dW = np.dot(liste_couches[i - 1].T, delta[i])  # Gradient pour les poids
        dB = np.sum(delta[i], axis=0, keepdims=True)  # Gradient pour les biais
        liste_poids[i - 1] -= alpha * dW  # Mise à jour des poids
        liste_biais[i - 1] -= alpha * dB  # Mise à jour des biais
    return liste_poids, liste_biais  # Retourne les poids et biais mis à jour


###############################################
### Fonctions de test du réseau de neurones ###
###############################################
def test_reseau_neurone_unique(image, etiquette, liste_W, liste_B):
    """
    Teste le réseau de neurones sur un exemple donné.
    X_data : matrice d'entrée (taille : nombre d'exemples x input_size)
    Y_data : vecteur des étiquettes (taille : nombre d'exemples)
    liste_W : liste des matrices des poids pour chaque couche
    liste_B : liste des vecteurs des biais pour chaque couche
    Retourne la sortie du réseau de neurones pour l'exemple donné.
    """

    nb_couches = len(liste_W) + 1  # Nombre total de couches (entrée + cachées + sortie)
    couche = image  # Couche d'entrée
    for i in range(nb_couches - 1):
        Z = somme_ponderee(couche, liste_W[i], liste_B[i])  # Somme pondérée pour la couche actuelle
        couche = fonction_activation(Z)  # Application de la fonction d'activation sur la somme pondérée

    erreur = cout_mse(couche, one_hot(etiquette))  # Calcul de l'erreur entre la sortie du réseau et le vecteur d'étiquette
    return couche, erreur  # Retourne la sortie du réseau de neurones

def test_global_reseau_neurone(X_data, Y_data, liste_W, liste_B):
    """
    Teste le réseau de neurones sur l'ensemble des données de test.
    X_data : matrice d'entrée (taille : nombre d'exemples x input_size)
    Y_data : vecteur des étiquettes (taille : nombre d'exemples)
    liste_W : liste des matrices des poids pour chaque couche
    liste_B : liste des vecteurs des biais pour chaque couche
    Retourne le taux de réussite du réseau de neurones sur l'ensemble des données de test.
    """
    nb_couches = len(liste_W) + 1  # Nombre total de couches (entrée + cachées + sortie)
    nb_images_test = X_data.shape[0]  # Nombre d'images dans les données de test
    nb_reussites = 0  # Compteur pour le nombre de réussites
    erreur_moyenne = 0  # Compteur pour la moyenne des erreurs

    for index_image in range(nb_images_test):
        prediction, erreur = test_reseau_neurone_unique(X_data[index_image:index_image+1], Y_data[index_image:index_image+1], liste_W, liste_B)  # Test du réseau de neurones sur l'image d'index index_image
        if prediction_values(prediction) == Y_data[index_image]:
            nb_reussites += 1
        erreur_moyenne += erreur

    taux_reussite = nb_reussites / nb_images_test * 100  # Calcul du taux de réussite en pourcentage
    erreur_moyenne /= nb_images_test  # Calcul de l'erreur moyenne
    return taux_reussite, erreur_moyenne  # Retourne le taux de réussite et l'erreur moyenne du réseau de neurones sur l'ensemble des données de test

def prediction_values(prediction_vector):
    """
    Transforme la sortie du réseau de neurones en prédiction de chiffre.
    prediction_vector : vecteur de sortie du réseau de neurones (taille : 1 x 10)
    Retourne l'indice du maximum, qui correspond au chiffre prédit.
    """
    return int(np.argmax(prediction_vector))














