import numpy as np

def normalize(vectors):
    normalized_vectors = []
    for v in vectors:
        norm = np.sqrt(sum([x ** 2 for x in v]))
        normalized_v = [x / norm for x in v]
        normalized_vectors.append(normalized_v)
    return np.array(normalized_vectors)