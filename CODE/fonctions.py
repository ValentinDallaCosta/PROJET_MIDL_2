import numpy as np

# Fonction d'activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fonction_activation(matrix):
    """
    Applique la fonction sigmoïde à chaque élément de la matrice.
    matrix : matrice d'entrée
    """
    return 1 / (1 + np.exp(-matrix))


########  Fonction Somme pondérée ########
def somme_ponderee(W,A,B):
    """
    Calcule la somme pondérée des entrées.
    W : matrice des poids (taille : input_size x 10)
    A : matrice des activations (taille : nombre d'exemples x input_size)
    B : vecteur des biais (taille : 1 x 10)
    """
    return np.average(np.dot(A, W) + B, axis=0)  # Somme pondérée pour chaque exemple


######## Fonction de coût ########
def cout(X, Y):
    """
    Fonction de coût pour le réseau de neurones.
    X : matrice d'entrée 
    Y : vecteur des étiquettes (taille : nombre d'exemples)
    """
    cout = np.average((X-Y)**2)
    return cout

######## Gradiant négatif de la fonction de coût #######
def negative_gradient(X, Y):
    """
    Calcule le gradient négatif de la fonction de coût.
    X : matrice d'entrée 
    Y : vecteur des étiquettes (taille : nombre d'exemples)
    """
    