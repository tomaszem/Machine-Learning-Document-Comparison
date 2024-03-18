#  TaggedDocument struktura používaná k označení dokumentů jedinečnými identifikátory
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
#  funkce pro rozdělení textu na jednotlivá slova (tokeny)
from nltk.tokenize import word_tokenize

filename = 'data/docs.txt'

with open(filename, 'r', encoding='utf-8') as file:
    documents = file.read().split('\n')  # každý dokument je na novém řádku

# preproces documentů
# TaggedDocument vytváří objekt, který má dva atributy: words, což je seznam tokenů dokumentu
# tags je seznam obsahující unikátní identifikátor dokumentu
tagged_data = [TaggedDocument(words=word_tokenize(doc.lower()), tags=[str(i)]) for i, doc in enumerate(documents) if doc.strip()]


# trénování Doc2vec modelu
# vector_size určuje velikost vektorů, které reprezentují dokumenty
# min_count říká modelu, aby ignoroval všechna slova s celkovým počtem výskytů menším než 2
# epocha určuje, kolikrát se mají data předložit modelu během trénování
model = Doc2Vec(vector_size=20, min_count=2, epochs=50)
model.build_vocab(tagged_data)
model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)


# model.infer_vector() je funkce, která přijímá seznam slov dokumentu
# a vrací vektorovou reprezentaci tohoto dokumentu podle naučeného modelu
document_vectors = [model.infer_vector(word_tokenize(doc.lower())) for doc in documents]


for i, doc in enumerate(documents):
    print(f"Document {i + 1}: {doc}")
    print(f"Vector: {document_vectors[i]}\n")