#fonctions pour trouver la base de données des images apprentissage 
import urllib.request
import gzip
import matplotlib.pyplot as plt
import numpy as np
import os

def charger_banque_apprentissage():
    url_image_apprentissage = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
    url_etiquettes_apprentissage = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"

    dossier_destination = "BANQUE_IMAGES"
    os.makedirs(dossier_destination, exist_ok=True)

    # Construction des chemins complets vers le dossier cible
    fichier_etiquettes = os.path.join(dossier_destination, "etiquettes_apprentissage.gz")
    fichier_images = os.path.join(dossier_destination, "images_apprentissage.gz")
    
    #telechargement des étiquettes cad une matrice ligne avec xi = 0 si pas le chiffre de l image 1 sinon
    #dans le dossier etiquettes_apprentissage.gz
    if not os.path.exists(fichier_etiquettes):
        print("Téléchargement des étiquettes de apprentissage ")
        urllib.request.urlretrieve(url_etiquettes_apprentissage, fichier_etiquettes)
    else:
        print("Vous avez deja les étiquettes de apprentissage")

    #telechargement des images du apprentissage qu on va transformer en matrice 784*10000 
    #dans le dossier images_apprentissage.gz
    if not os.path.exists(fichier_images):
        print("Téléchargement des images de apprentissage ")
        urllib.request.urlretrieve(url_image_apprentissage, fichier_images)
    else:
        print("Vous avez deja les images de apprentissage")


def afficher_image_banque(index):
    """
    Affiche l'image située à la position 'index' dans la banque d apprentissage,
    ainsi que son vrai chiffre associé (y). Fait par gemini
    """
    dossier_destination = "BANQUE_IMAGES"
    fichier_etiquettes = os.path.join(dossier_destination, "etiquettes_apprentissage.gz")
    fichier_images = os.path.join(dossier_destination, "images_apprentissage.gz")

    with gzip.open(fichier_etiquettes, 'rb') as f:
        y_apprentissage = np.frombuffer(f.read(), np.uint8, offset=8)

    # Lecture des images et mise sous forme de matrice (10000 images, 784 pixels)
    with gzip.open(fichier_images, 'rb') as f:
        X_apprentissage = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)
        X_apprentissage = X_apprentissage / 255.0  # Normalisation entre 0.0 et 1.0

    # 1. On extrait la ligne de 784 pixels et on la retransforme en grille 28x28
    image_2d = X_apprentissage[index].reshape(28, 28)
    vrai_chiffre = y_apprentissage[index]

    # 2. Affichage avec Matplotlib
    plt.imshow(image_2d, cmap='gray') # 'gray' pour l'afficher en noir et blanc
    plt.title(f"Image n°{index} - Vrai chiffre (Étiquette) : {vrai_chiffre}")
    plt.axis('off') # Cache les axes X et Y
    plt.show()

# Exemple : Afficher la 1re image (index 0) et la 42e image (index 41)
charger_banque_apprentissage()
afficher_image_banque(0)
afficher_image_banque(1)

