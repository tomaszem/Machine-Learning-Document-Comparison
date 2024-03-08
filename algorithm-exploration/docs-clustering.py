from PyPDF2 import PdfReader
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plot
import numpy as np
import re
import os
from collections import defaultdict

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

from pca_implementation import pca
from normalize_vectors import normalize
from concurrent.futures import ThreadPoolExecutor
import time
import yaml
from scipy.spatial import distance_matrix
from ravendb import DocumentStore
from data_storage import DataStorage

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


# Function to load stopwords from a YAML file
def load_stopwords(file_path):
    with open(file_path, 'r') as file:
        stopwords = yaml.safe_load(file)
    return set(stopwords)


# Path to your stopwords YAML file (update this to the correct path)
stopwords_file_path = 'config/stopwords.yaml'

# Load the stopwords
stopwords_set = load_stopwords(stopwords_file_path)


# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove all non-alphanumeric characters
    text = re.sub(r'\W+', ' ', text)
    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords_set]
    return ' '.join(filtered_words)


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
    # Initialize the TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()

    # Fit and transform the texts to a TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

    # Get feature names to locate columns in matrix
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Apply custom weights
    for word, weight in word_weights.items():
        if word in feature_names:
            # Get the column index of the word
            col_index = np.where(feature_names == word)[0][0]
            # Apply the custom weight by multiplying the column by the weight
            tfidf_matrix[:, col_index] *= weight

    # Normalize the tf-idf matrix rows after applying custom weights
    from sklearn.preprocessing import normalize
    tfidf_matrix_weighted = normalize(tfidf_matrix, axis=1, norm='l2')

    return tfidf_matrix_weighted.toarray()


# Using custom weights for vectorization
custom_weights = {"java": 12, "javascript": 5, "python": 9, "algebra": 3, "numerical": 3, "sql": 10}
custom_vectors = custom_vectorization(texts, custom_weights)

# Convert dense matrix to ndarray
#custom_vectors_array = np.asarray(custom_vectors)

# PCA reduction to 2 dimensions
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(custom_vectors)
print(reduced_vectors[0])

# DBSCAN clustering
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

# Load configuration
# with open('config/db-config.yaml', 'r') as config_file:
#     config = yaml.safe_load(config_file)
#     ravendb_config = config['ravendb']
#
# store = DocumentStore(urls=[ravendb_config['url']], database=ravendb_config['database'])
# store.initialize()

# Create an instance of DataStorage
# data_storage = DataStorage(store)

# Assume dist_matrix_2d and filenames are defined earlier in your main.py

# dist_matrix = distance_matrix(reduced_vectors, reduced_vectors)

# Use DataStorage instance to store distances
# data_storage.store_document_distances(dist_matrix, filenames)
