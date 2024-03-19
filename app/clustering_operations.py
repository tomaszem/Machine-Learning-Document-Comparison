from sklearn.decomposition import PCA
from app.extract_text import load_texts_from_pdfs_batched
from app.tf_idf_algorithm import custom_vectorization
from sklearn.cluster import DBSCAN
from umap import UMAP


def perform_clustering():
    # PCA reduction to 2 dimensions
    BATCH_SIZE = 5
    texts, filenames = load_texts_from_pdfs_batched(BATCH_SIZE)

    # custom_weights = {"algebra": 3, "numerical": 3, "sql": 10}
    custom_weights = {}
    custom_vectors = custom_vectorization(texts, custom_weights)

    # UMAP for dimensionality reduction
    #umap = UMAP(n_components=2)
    #reduced_vectors = umap.fit_transform(custom_vectors)

    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(custom_vectors)

    # DBSCAN clustering
    # PCA 0.05; 0.08
    dbscan = DBSCAN(eps=0.08, min_samples=2)
    clusters = dbscan.fit_predict(reduced_vectors)

    return filenames, reduced_vectors, clusters
