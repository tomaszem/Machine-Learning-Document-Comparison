from PyPDF2 import PdfReader
import numpy as np
import os
import re
import math
import matplotlib.pyplot as plot
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN
from pca_implementation import pca
from normalize_vectors import normalize

# Funkce pro načtení textu z PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() if page.extract_text() else ''
        return text


# Funkce pro předzpracování textu
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text


# Načtení textů z PDF souborů
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


# Funkce pro výpočet TF (Term Frequency)
def compute_tf(text):
    tf_text = Counter(text.split())
    for word in tf_text:
        tf_text[word] = tf_text[word] / float(len(text.split()))
    return tf_text


# Funkce pro výpočet IDF (Inverse Document Frequency)
def compute_idf(documents):
    idf_dict = defaultdict(lambda: 0)
    N = len(documents)

    for document in documents:
        for word in set(document.split()):
            idf_dict[word] += 1

    for word in idf_dict:
        idf_dict[word] = math.log(N / float(idf_dict[word]))

    return idf_dict


# Vektorizace textů pomocí TF-IDF
def tfidf_vectorization(texts):
    documents_tf = [compute_tf(text) for text in texts]
    idf = compute_idf(texts)

    tfidf_documents = []
    for tf in documents_tf:
        tfidf = {}
        for word, tf_value in tf.items():
            tfidf[word] = tf_value * idf[word]
        tfidf_documents.append(tfidf)

    # Vytvoření matice TF-IDF
    unique_words = set(word for document in texts for word in document.split())
    tfidf_matrix = []
    for tfidf in tfidf_documents:
        tfidf_matrix.append([tfidf.get(word, 0) for word in unique_words])

    return np.array(tfidf_matrix)


folder_path = 'documents'
texts, filenames = load_texts_from_pdfs(folder_path)
tfidf_vectors = tfidf_vectorization(texts)
normalized_vectors = normalize(tfidf_vectors)

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
plot.title('Vizualizace výsledků TF-IDF')
plot.show()
