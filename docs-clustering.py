from PyPDF2 import PdfReader
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plot
import numpy as np
import re
import os
from collections import defaultdict
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
    # Nahrazení všech nealfanumerických znaků
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

# Načtení a předzpracování textů
folder_path = 'documents'
texts, filenames = load_texts_from_pdfs(folder_path)
print (texts[5]);

def build_vocabulary(texts):
    # Vytvoří slovník všech jedinečných slov v seznamu textů
    vocabulary = set()
    for text in texts:
        words = text.split()
        vocabulary.update(words)
    return list(vocabulary)

def vectorize_text(text, vocabulary, word_weights):
    # Vytvoří vektor pro daný text na základě slovníku a váh slov
    word_count = defaultdict(int)
    for word in text.split():
        word_count[word] += 1

    # Nulový vektor o velikosti slovníku
    vector = np.zeros(len(vocabulary))
    for i, word in enumerate(vocabulary):
        vector[i] = word_count[word] * word_weights.get(word, 1)  # Použití váhy slova, pokud existuje
    return vector

def custom_vectorization(texts, word_weights={}):
    # Vlastní vektorizace textů s možností definovat váhy pro slova
    vocabulary = build_vocabulary(texts)
    vectors = np.array([vectorize_text(text, vocabulary, word_weights) for text in texts])
    # Normalizace vektorů
    return normalize(vectors)


# Příklad vlastních vah pro určitá slova
custom_weights = {"java": 12, "javascript": 5, "python": 19, "algebra": 3, "numerical": 3, "sql": 30}

# Použití vlastní vektorizace
custom_vectors = custom_vectorization(texts, custom_weights)

# PCA redukce
reduced_vectors = pca(custom_vectors, 2)

# DBSCAN clusterizace
dbscan = DBSCAN(eps=0.5, min_samples=2)
clusters = dbscan.fit_predict(reduced_vectors)

# Vizualizace s názvy dokumentů
plot.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis')
for i, filename in enumerate(filenames):
    plot.text(reduced_vectors[i, 0], reduced_vectors[i, 1], filename, fontsize=9)
plot.xlabel('Komponenta X')
plot.ylabel('Komponenta Y')
plot.title('Vizualizace výsledků clusterizace pomocí DBSCAN')
plot.show()
