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
            "id": str(uuid.uuid4()),  # Generate a new UUID for each document
            "filename": filename,
            "x": float(reduced_vectors[i, 0]),  # Ensure JSON serializability
            "y": float(reduced_vectors[i, 1]),
            "cluster": int(clusters[i])
        }
        for i, filename in enumerate(filenames)
    ]

    return json_data
