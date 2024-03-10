from sklearn.decomposition import PCA
from app.extract_text import load_texts_from_pdfs_batched
from app.tf_idf_algorithm import custom_vectorization
from sklearn.cluster import DBSCAN


def perform_clustering():
    # PCA reduction to 2 dimensions
    BATCH_SIZE = 5
    texts, filenames = load_texts_from_pdfs_batched(BATCH_SIZE)

    custom_weights = {"java": 12, "javascript": 5, "python": 9, "algebra": 3, "numerical": 3, "sql": 10}
    custom_vectors = custom_vectorization(texts, custom_weights)
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(custom_vectors)

    # DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    clusters = dbscan.fit_predict(reduced_vectors)

    return filenames, reduced_vectors, clusters
