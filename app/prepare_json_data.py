import uuid

"""
Prepare data for JSON serialization.

Parameters:
- filenames: A list of filenames or IDs corresponding to the documents.
- reduced_vectors: A 2D numpy array where each row corresponds to the reduced vector of a document.
- clusters: A list or array of cluster labels for each document.

Returns:
- JSON data
"""


def prepare_json_data(filenames, reduced_vectors, clusters):
    json_data = [
        {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "x": float(reduced_vectors[i, 0]),
            "y": float(reduced_vectors[i, 1]),
            "cluster": int(clusters[i])
        }
        for i, filename in enumerate(filenames)
    ]

    return json_data


def prepare_json_data_v2(filenames, reduced_vectors_3d, initial_clusters, reduced_vectors_2d, final_clusters):
    json_data = [
        {
            "id": str(uuid.uuid4()),
            "filename": filename,
            "x_3d": float(reduced_vectors_3d[i, 0]),
            "y_3d": float(reduced_vectors_3d[i, 1]),
            "z_3d": float(reduced_vectors_3d[i, 2]),
            "x_2d": float(reduced_vectors_2d[i][0]),
            "y_2d": float(reduced_vectors_2d[i][1]),
            "initial_cluster": int(initial_clusters[i]),
            "final_cluster": int(final_clusters[i])
        }
        for i, filename in enumerate(filenames)
    ]

    return json_data
