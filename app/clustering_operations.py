import yaml
from sklearn.decomposition import PCA
from app.extract_text import load_texts_from_pdfs_batched
from app.tf_idf_algorithm import custom_vectorization
from sklearn.cluster import DBSCAN
# from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from app.config.constants import BATCH_SIZE
from app.optimal_number_of_clusters import number_of_clusters
from app.config.constants import CONFIG_PATH


def perform_clustering():
    texts, filenames = load_texts_from_pdfs_batched(BATCH_SIZE)

    with open(CONFIG_PATH, 'r') as file:
        config = yaml.safe_load(file)

    eps_num = config['dbscan']['eps']
    n_cluster_conf = config['agglomerative_clusters']['n']

    custom_weights = {}
    custom_vectors = custom_vectorization(texts, custom_weights)

    # UMAP for dimensionality reduction
    # umap = UMAP(n_components=2)
    # reduced_vectors = umap.fit_transform(custom_vectors)

    # DBSCAN
    # dbscan = DBSCAN(eps=0.08, min_samples=2)
    # initial_clusters = dbscan.fit_predict(reduced_vectors)

    # First stage
    pca_3d = PCA(n_components=3)
    reduced_vectors_3d = pca_3d.fit_transform(custom_vectors)

    clusters_num = n_cluster_conf if n_cluster_conf is not None else number_of_clusters(reduced_vectors_3d)
    agglomerative_clustering = AgglomerativeClustering(n_clusters=clusters_num, linkage='ward')
    initial_clusters = agglomerative_clustering.fit_predict(reduced_vectors_3d)

    # Prepare for Second stage clustering
    final_clusters = []
    reduced_vectors_2d = []
    pca_2d = PCA(n_components=2)
    for cluster_id in set(initial_clusters):
        cluster_indices = [i for i, x in enumerate(initial_clusters) if x == cluster_id]
        cluster_vectors = custom_vectors[cluster_indices]

        # Reduce to 2D within each cluster and re-cluster
        reduced_vectors_2d_temp = pca_2d.fit_transform(cluster_vectors)
        reduced_vectors_2d.extend(reduced_vectors_2d_temp)

        dbscan_refined = DBSCAN(eps=eps_num, min_samples=2)
        refined_clusters = dbscan_refined.fit_predict(reduced_vectors_2d_temp)

        # Ensures that the new set of cluster IDs does not overlap with any existing IDs.
        final_clusters.extend(refined_clusters + max(final_clusters + [0]) + 1 if final_clusters else refined_clusters)

    return filenames, reduced_vectors_3d, initial_clusters, reduced_vectors_2d, final_clusters, custom_vectors
