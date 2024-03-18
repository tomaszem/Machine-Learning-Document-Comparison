import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


def vectorize_text(text, vocabulary, word_weights):
    word_count = defaultdict(int)
    for word in text.split():
        word_count[word] += 1

    vector = np.zeros(len(vocabulary))
    for i, word in enumerate(vocabulary):
        vector[i] = word_count[word] * word_weights.get(word, 1)
    return vector


def custom_vectorization(texts, word_weights={}):
    # Initialize the TF-IDF
    tfidf_vectorized = TfidfVectorizer()

    # Fit and transform the texts to a TF-IDF matrix
    tfidf_matrix = tfidf_vectorized.fit_transform(texts)

    # Get feature names to locate columns in matrix
    feature_names = tfidf_vectorized.get_feature_names_out()

    # Apply custom weights
    for word, weight in word_weights.items():
        if word in feature_names:
            # Get the column index of the word
            col_index = np.where(feature_names == word)[0][0]
            # Apply the custom weight by multiplying the column by the weight
            tfidf_matrix[:, col_index] *= weight

    # Normalize the tf-idf matrix rows after applying custom weights
    tfidf_matrix_weighted = normalize(tfidf_matrix, axis=1, norm='l2')

    return tfidf_matrix_weighted.toarray()
