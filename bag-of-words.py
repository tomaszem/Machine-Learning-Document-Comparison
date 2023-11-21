from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plot
from collections import Counter

# Example documents
documents = []
with open('data.txt', 'r', encoding='utf-8') as file:
    for line in file:
        documents.append(line.strip())

unique_words = set(" ".join(documents).lower().split())
word_to_index = {word: i for i, word in enumerate(unique_words)}

# Bag of Words matice
def create_bow_matrix(sentences, word_to_index):
    # Inicializace nulové matice s rozměry: počet vět x počet jedinečných slov
    matrix = np.zeros((len(sentences), len(word_to_index)))
    # Iterace přes všechny věty a jejich indexy
    for i, sentence in enumerate(sentences):
        # Vytvoření počtu výskytů slov ve větě, převedení na malá písmena a rozdělení na slova
        words = Counter(sentence.lower().split())
        # Iterace přes všechna slova v aktuální větě
        for word in words:
            # Kontrola, zda je slovo ve slovníku word_to_index
            if word in word_to_index:
                # Nastavení hodnoty 1 v matici na příslušném místě, pokud je slovo ve slovníku
                matrix[i, word_to_index[word]] = 1
    return matrix


# BoW matice
sentence_vectors = create_bow_matrix(documents, word_to_index)

# Redukce dimenze - PCA
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(sentence_vectors)

#  DBSCAN odděluje vysokohustotní clustery od nízkohustotního šumu
#  umožňuje identifikaci clusterů různých tvarů a velikostí bez
#  nutnosti předem specifikovat počet clusterů
dbscan = DBSCAN(eps=1.0, min_samples=2)
clusters = dbscan.fit_predict(reduced_vectors)

# Results
plot.figure(figsize=(8, 6))
plot.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=clusters, cmap='viridis')
plot.title("Sentence Clustering with Bag of Words and DBSCAN")
plot.xlabel("PCA Component 1")
plot.ylabel("PCA Component 2")

for i, sentence in enumerate(documents):
    plot.annotate(sentence, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

plot.show()
