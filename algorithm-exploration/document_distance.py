class DocumentComparison:
    def __init__(self, target_document, distance):
        self.target_document = target_document
        self.distance = distance

class DocumentDistance:
    def __init__(self, document_id, comparisons):
        self.document_id = document_id
        self.comparisons = comparisons
