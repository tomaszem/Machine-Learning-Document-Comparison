from PyPDF2 import PdfReader
import numpy as np
import re
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from pca_implementation import pca
from normalize_vectors import normalize

# Funkce pro načtení textu z PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

# Funkce pro předzpracování textu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Načtení textů z PDF souborů a uložení jejich názvů
def load_texts_from_pdfs(folder_path):
    texts = []
    filenames = []
    for file in os.listdir(folder_path):
        if file.endswith('.pdf'):
            file_path = os.path.join(folder_path, file)
            text = extract_text_from_pdf(file_path)
            cleaned_text = preprocess_text(text)
            texts.append(cleaned_text)
            filenames.append(file)
    return texts, filenames

# Vytvoření slovníku všech jedinečných slov v seznamu textů
def build_vocabulary(texts):
    vocabulary = set()
    for text in texts:
        words = text.split()
        vocabulary.update(words)
    return list(vocabulary)

# Vektorizace textu
def binary_vectorize_text(text, vocabulary):
    # Vytvoření binárního vektoru pro daný text na základě slovníku
    vector = np.zeros(len(vocabulary), dtype=int)
    words = text.split()
    for word in words:
        if word in vocabulary:
            index = vocabulary.index(word)
            vector[index] = 1
    return vector

# Upravená funkce pro vektorizaci
def custom_vectorization(texts):
    vocabulary = build_vocabulary(texts)
    vectors = np.array([binary_vectorize_text(text, vocabulary) for text in texts])
    return normalize(vectors), vocabulary

# Načtení a předzpracování textů
folder_path = '../documents'  # Název složky s dokumenty
texts, filenames = load_texts_from_pdfs(folder_path)

# Použití vlastní vektorizace
vectors, vocabulary = custom_vectorization(texts)

reduced_vectors = pca(vectors, 2)

# DBSCAN clusterizace
dbscan = DBSCAN(eps=0.5, min_samples=2)
clusters = dbscan.fit_predict(reduced_vectors)

# Vizualizace s názvy dokumentů
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis')
for i, filename in enumerate(filenames):
    plt.text(reduced_vectors[i, 0], reduced_vectors[i, 1], filename, fontsize=9)
plt.xlabel('Komponenta X')
plt.ylabel('Komponenta Y')
plt.title('Vizualizace výsledků Binary Encoding')
plt.show()
