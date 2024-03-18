from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plot

# Example sentences
sentences = [
    "The quick brown fox jumps over the lazy dog",
    "A quick brown dog outpaces a fast fox",
    "The dog barks loudly at the sun",
    "A lazy fox sleeps all day",
    "The sun shines brightly"
]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)

# TSNE pracuje tak, že nejprve převede vzdálenosti mezi datovými body v původním vysokodimenzionálním
# prostoru na pravděpodobnostní rozdělení, které reprezentuje podobnosti mezi body.
# Poté vytváří nízkodimenzionální mapu dat, kde se snaží zachovat podobné pravděpodobnostní rozdělení, což
# vede k zachování struktury původních dat. Nakonec t-SNE iterativně upravuje pozice bodů v nízkodimenzionálním
# prostoru tak, aby minimalizoval rozdíly mezi dvěma pravděpodobnostními rozděleními, čímž umožňuje efektivní
# vizualizaci klastrů a vzorců ve složitých datech.

tsne = TSNE(n_components=3, perplexity=3, n_iter=1000, random_state=42)
reduced_vectors_tsne = tsne.fit_transform(tfidf_matrix.toarray())

# Vytvoření 3D grafu
fig = plot.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(reduced_vectors_tsne[:, 0], reduced_vectors_tsne[:, 1], reduced_vectors_tsne[:, 2])

ax.set_title("Sentence Clustering with T-SNE in 3D")
ax.set_xlabel("T-SNE Component 1")
ax.set_ylabel("T-SNE Component 2")
ax.set_zlabel("T-SNE Component 3")

# Přidání anotací
for i, sentence in enumerate(sentences):
    ax.text(reduced_vectors_tsne[i, 0], reduced_vectors_tsne[i, 1], reduced_vectors_tsne[i, 2], sentence)

plot.tight_layout()
plot.show()
