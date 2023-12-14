from PyPDF2 import PdfReader
import numpy as np
import os
import re
from collections import defaultdict
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plot
from pca_implementation import pca
from normalize_vectors import normalize

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

# Načtení textů z více PDF souborů a vrácení názvů souborů
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

# Funkce pro vytvoření slovníku
def build_vocabulary(texts):
    vocabulary = set()
    for text in texts:
        words = text.split()
        vocabulary.update(words)
    return list(vocabulary)

# Funkce pro vektorizaci textů pomocí počtu výskytů slov
def word_count_vectorize(texts, vocabulary):
    vectors = []
    for text in texts:
        word_count = defaultdict(int)
        for word in text.split():
            word_count[word] += 1
        vector = [word_count[word] for word in vocabulary]
        vectors.append(vector)
    return np.array(vectors)

# Načtení a předzpracování textů
folder_path = 'documents'
texts, filenames = load_texts_from_pdfs(folder_path)

# Vytvoření slovníku a vektorizace textů
vocabulary = build_vocabulary(texts)
word_count_vectors = word_count_vectorize(texts, vocabulary)

# Normalizace vektorů
normalized_vectors = normalize(word_count_vectors)

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
plot.title('Vizualizace výsledků Word Count Vectorizer')
plot.show()
