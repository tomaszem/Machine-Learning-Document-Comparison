from PyPDF2 import PdfReader
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plot
import numpy as np
import re
import os
from collections import defaultdict
from pca_implementation import pca
from normalize_vectors import normalize
from concurrent.futures import ThreadPoolExecutor
import time


# Constant - definition of batch size
BATCH_SIZE = 5

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

# Function for text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    return text

# Processing files in a batch
def process_files_batch(files_batch):
    texts = []
    filenames = []
    for file in files_batch:
        file_path = os.path.join(folder_path, file)
        text = extract_text_from_pdf(file_path)
        cleaned_text = preprocess_text(text)
        texts.append(cleaned_text)
        filenames.append(file)
    return texts, filenames

# Loading texts from multiple PDF files in batches
def load_texts_from_pdfs_batched(folder_path, batch_size):
    all_texts = []
    all_filenames = []
    files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]

    for i in range(0, len(files), batch_size):
        files_batch = files[i:i + batch_size]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_files_batch, [file]) for file in files_batch]
            for future in futures:
                texts, filenames = future.result()
                all_texts.extend(texts)
                all_filenames.extend(filenames)

    return all_texts, all_filenames

# Start timer
start_time = time.time()
# Loading and preprocessing texts
folder_path = 'documents'
texts, filenames = load_texts_from_pdfs_batched(folder_path, BATCH_SIZE)

# Creating a vocabulary and vectorizing texts
def build_vocabulary(texts):
    vocabulary = set()
    for text in texts:
        words = text.split()
        vocabulary.update(words)
    return list(vocabulary)

def vectorize_text(text, vocabulary, word_weights):
    word_count = defaultdict(int)
    for word in text.split():
        word_count[word] += 1

    vector = np.zeros(len(vocabulary))
    for i, word in enumerate(vocabulary):
        vector[i] = word_count[word] * word_weights.get(word, 1)
    return vector

def custom_vectorization(texts, word_weights={}):
    vocabulary = build_vocabulary(texts)
    def vectorize(text):
        return vectorize_text(text, vocabulary, word_weights)

    with ThreadPoolExecutor() as executor:
        vectors = list(executor.map(vectorize, texts))

    return normalize(np.array(vectors))

# Using custom weights for vectorization
custom_weights = {"java": 12, "javascript": 5, "python": 19, "algebra": 3, "numerical": 3, "sql": 30}
custom_vectors = custom_vectorization(texts, custom_weights)

# PCA reduction and DBSCAN clustering
reduced_vectors = pca(custom_vectors, 2)
dbscan = DBSCAN(eps=0.5, min_samples=2)
clusters = dbscan.fit_predict(reduced_vectors)

# Visualization of results
plot.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis')
for i, filename in enumerate(filenames):
    plot.text(reduced_vectors[i, 0], reduced_vectors[i, 1], filename, fontsize=9)
plot.xlabel('Component X')
plot.ylabel('Component Y')
plot.title('Visualization of Clustering Results Using DBSCAN')
plot.show()

# Total execution time
end_time = time.time()
total_time = end_time - start_time

print(f"Total execution time: {total_time:.2f} sec.")
