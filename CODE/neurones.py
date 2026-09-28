#fonctions pour trouver la base de données des images test 
import urllib.request
import gzip

def charger_banque_test():
    url_image_test = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz"
    url_etiquettes_test = "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"

    #telechargement des étiquettes cad une matrice ligne avec xi = 0 si pas le chiffre de l image 1 sinon
    #dans le dossier etiquettes_test.gz
    urllib.request.urlretrieve(url_etiquettes_test, "etiquettes_test.gz")

    #telechargement des images du test qu on va transformer en matrice 784*10000 
    #dans le dossier images_test.gz
    urllib.request.urlretrieve(url_image_test,"images_test.gz")

charger_banque_test()