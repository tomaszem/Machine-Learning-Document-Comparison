def calculate_euclidean_distances(dist_matrix, filenames):
    comparisons = []
    for i, _ in enumerate(filenames):
        comparisons_row = [(dist, other_file)
                           for j, (dist, other_file) in enumerate(zip(dist_matrix[i], filenames))
                           if i != j]
        comparisons_row.sort(key=lambda x: x[0])
        comparisons.append(comparisons_row)
    return comparisons
