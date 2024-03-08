from document_distance import DocumentDistance, DocumentComparison
from distance_calculator import calculate_euclidean_distances

class DataStorage:
    def __init__(self, store):
        self.store = store

    def store_document_distances(self, dist_matrix, filenames):
        with self.store.open_session() as session:
            comparisons = calculate_euclidean_distances(dist_matrix, filenames)

            for i, filename in enumerate(filenames):
                document_comparisons = [DocumentComparison(target_document=other_file, distance=dist)
                                        for dist, other_file in comparisons[i]]
                document_distances = DocumentDistance(document_id=filename, comparisons=document_comparisons)
                session.store(document_distances)

            session.save_changes()
