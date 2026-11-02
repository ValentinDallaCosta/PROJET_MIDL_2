import gzip
import matplotlib.pyplot as plt
import numpy as np

#ici on va creer notre reseau de neuronnes en nous basant sur la formule W = (X.T * X)^-1 * X.T * Y

def reseaux( X,Y):
    X_transpose = X.T
    inverse = np.linalg.inv(X_transpose * X)
    W = inverse * X_transpose * Y
    return W


