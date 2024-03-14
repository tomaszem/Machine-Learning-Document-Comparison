import chromadb
from chromadb.utils import embedding_functions

default_ef = embedding_functions.DefaultEmbeddingFunction()
client = chromadb.HttpClient(host='localhost', port=8000)


client.delete_collection(name="my_collection")