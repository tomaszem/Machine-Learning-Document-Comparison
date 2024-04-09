import numpy as np


# Implementace PCA algoritmu
def pca(X, num_components):
    # Vypočítáme průměr dat pro každý atribut
    mean = np.mean(X, axis=0)

    # Odečteme průměr od všech dat, abychom měli data centrována
    centered_X = X - mean

    # Výpočet kovarianční matice
    cov_matrix = np.cov(centered_X, rowvar=False)

    # Výpočet vlastních hodnot a vlastních vektorů kovarianční matice
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Seřadíme vlastní vektory sestupně podle hodnoty vlastních hodnot
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Vybereme hlavní komponenty
    top_eigenvectors = eigenvectors[:, :num_components]

    # Projekce centrovaných dat
    reduced_X = np.dot(centered_X, top_eigenvectors)

    return reduced_X
