from PyPDF2 import PdfReader
import os
import re
import numpy as np
import matplotlib.pyplot as plot
from sklearn.cluster import DBSCAN
from normalize_vectors import normalize
from pca_implementation import pca

# Funkce pro načtení textu z PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Funkce pro předzpracování textu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Načtení textů z více PDF souborů
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

folder_path = '../documents'
texts, filenames = load_texts_from_pdfs(folder_path)

# Bag of Words vektorizace
def bag_of_words_vectorization(texts):
    vocabulary = set()
    for text in texts:
        vocabulary.update(text.split())
    vocabulary = sorted(vocabulary)

    vectors = np.zeros((len(texts), len(vocabulary)))
    for i, text in enumerate(texts):
        for word in text.split():
            if word in vocabulary:
                vectors[i][vocabulary.index(word)] += 1
    return vectors, vocabulary

# Použití Bag of Words vektorizace
bow_vectors, vocabulary = bag_of_words_vectorization(texts)

# Normalizace vektorů
normalized_vectors = normalize(bow_vectors)

# PCA redukce
reduced_vectors = pca(normalized_vectors, 2)

# DBSCAN clusterizace
dbscan = DBSCAN(eps=0.5, min_samples=2)
clusters = dbscan.fit_predict(reduced_vectors)

# Vizualizace s názvy dokumentů
plot.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis')
for i, filename in enumerate(filenames):
    plot.text(reduced_vectors[i, 0], reduced_vectors[i, 1], filename, fontsize=9)
plot.xlabel('Komponenta X')
plot.ylabel('Komponenta Y')
plot.title('Vizualizace výsledků BoW algoritmu')
plot.show()
