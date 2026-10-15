import numpy as np

# Fonction d'activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def fonction_activation(matrix):
    """
    Applique la fonction sigmoïde à chaque élément de la matrice.
    matrix : matrice d'entrée
    """
    return 1 / (1 + np.exp(-matrix))


########  Fonction Somme pondérée ########
def somme_ponderee(A,W,B):
    """
    Calcule la somme pondérée des entrées.
    W : matrice des poids (taille : input_size x 10)
    A : matrice des activations (taille : nombre d'exemples x input_size)
    B : vecteur des biais (taille : 1 x 10)
    """
    return np.dot(A, W) + B  # Somme pondérée pour chaque exemple


######## Fonction de coût ########
def cout_mse(X, Y):
    """
    Fonction de coût pour le réseau de neurones.
    X : matrice d'entrée 
    Y : vecteur des étiquettes (taille : nombre d'exemples)
    """
    cout = 1/2 * np.sum((X-Y)**2)
    return cout

def one_hot_encode(y):
    """
    Transforme un entier en vecteur one-hot.
    y : entier représentant l'étiquette (0 à 9)
    Retourne un vecteur de taille 10 avec un 1 à la position de l'étiquette et des 0 ailleurs.
    """
    vecteur_etiquette = np.zeros(10)
    vecteur_etiquette[y] = 1
    return vecteur_etiquette