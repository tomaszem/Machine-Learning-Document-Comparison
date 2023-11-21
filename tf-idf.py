from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plot
import math

# Example sentences
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a fast fox",
    "The dog barks loudly at the sun",
    "A lazy fox sleeps all day",
    "The sun shines brightly"
]

# Vytvoří unikátní index pro každé slovo z dokumentů
unique_words = set(" ".join(sentences).lower().split())
word_to_index = {word: i for i, word in enumerate(unique_words)}

# Věty => embeddings
def sentence_to_vector(sentence, word_to_index, idf):
    words = sentence.lower().split() # jednotlivá slova ve větě
    word_count = len(words) # počet slov
    word_freq = {word: words.count(word) / word_count for word in words} # počet výskytů slova
    vector = np.zeros(len(word_to_index))
    for word in words:
        index = word_to_index[word]
        vector[index] = word_freq[word] * idf[word]
    return vector

def pca(X, num_components=2):
    # Vypočítáme průměr dat pro každý atribut
    mean = np.mean(X, axis=0)

    # Odečteme průměr od všech dat, abychom měli data centrována
    centered_X = X - mean

    # Výpočet kovarianční matice
    cov_matrix = np.cov(centered_X, rowvar=False)

    # Výpočet vlastních hodnot a vlastních vektorů kovarianční matice
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Seřadíme vlastní vektory sestupně podle hodnoty vlastních hodnot
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Vybereme první 'num_components' vlastních vektorů, což jsou hlavní komponenty
    top_eigenvectors = eigenvectors[:, :num_components]

    # Projekce zcentrovaných dat na hlavní komponenty
    reduced_X = np.dot(centered_X, top_eigenvectors)

    return reduced_X


# Výpočet IDF pro každé slovo
doc_count = len(sentences)

idf = {}
for word in unique_words:
    doc_freq = sum(word in sentence.lower().split() for sentence in sentences)
    idf[word] = math.log(doc_count / (1 + doc_freq))

# Každá věta je převedena na vektor pomocí vypočítaných IDF hodnot a TF (četnost termínu)
sentence_vectors = np.array([sentence_to_vector(sentence, word_to_index, idf) for sentence in sentences])

# Použití funkce PCA
reduced_vectors = pca(sentence_vectors)

# kmeans
kmeans = KMeans(n_clusters=3, n_init=10)
kmeans.fit(reduced_vectors)
clusters = kmeans.predict(reduced_vectors)

# Results
plot.figure(figsize=(8, 6))
plot.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis')
plot.title("Sentence Clustering with TF-IDF")
plot.xlabel("PCA Component 1")
plot.ylabel("PCA Component 2")

for i, sentence in enumerate(sentences):
    plot.annotate(sentence, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

plot.show()
