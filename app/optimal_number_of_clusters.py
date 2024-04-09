from sklearn.metrics import davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering

"""
Suggests the optimal number of clusters for Agglomerative Clustering
which is based on the Davies-Bouldin Index.

Parameters:
- reduced_vectors_3d: The embeddings reduced to three dimensions.
- min_clusters: Minimum number of clusters to try.
- max_clusters: Maximum number of clusters to try.

Returns:
- The optimal number of clusters.
"""


def number_of_clusters(reduced_vectors_3d, min_clusters=2, max_clusters=10):
    best_score = float('inf')
    best_n_clusters = 0

    for n_clusters in range(min_clusters, max_clusters + 1):
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        cluster_labels = agglomerative.fit_predict(reduced_vectors_3d)
        score = davies_bouldin_score(reduced_vectors_3d, cluster_labels)

        if score < best_score:
            best_score = score
            best_n_clusters = n_clusters

    return best_n_clusters
