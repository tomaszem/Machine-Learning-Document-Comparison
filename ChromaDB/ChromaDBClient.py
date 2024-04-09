import os
import fitz  # PyMuPDF
import chromadb
import numpy as np
from sklearn.decomposition import PCA
from app import pdf_info_extraction
import chromadb.utils.embedding_functions as emb_fn

default_ef = emb_fn.DefaultEmbeddingFunction()
client = chromadb.HttpClient(host='localhost', port=8000)

collection = client.get_or_create_collection(name="pdf_collection", embedding_function=default_ef)

pdf_testPath = None
texts = None


def update_collection_chromadb(pdf_directory):
    pdf_details = pdf_info_extraction.extract_details(pdf_directory)
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            metadata = pdf_details.get(filename)
            if metadata is not None:
                collection.upsert(
                    documents=[filename],
                    metadatas=[{
                        "title": metadata["title"],
                        "authors": metadata["authors"],
                        "abstract": metadata["abstract"],
                        "references": metadata["references"]
                    }],
                    ids=[filename]
                )


def extract_text_from_pdf(pdf_testPath):
    text = ''
    doc = fitz.open(pdf_testPath)
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text += page.get_text()
    return text


def perform_query(texts):
    results = collection.query(
        query_texts=[str(texts)],
        include=["documents", "distances", "metadatas"],
        n_results=4
    )
    print(results)


def get_collection():
    try:
        collection = client.get_collection(name="pdf_collection")
        collection = collection.get(include=["documents", "metadatas", "embeddings"])
        return collection
    except chromadb.exceptions.CollectionNotFound:
        return None


if __name__ == "__main__":
    pdf_directory = "compareDocuments"
    pdf_testPath = "../compareDocuments"

    pdf_files = [os.path.join(pdf_testPath, f) for f in os.listdir(pdf_testPath) if f.endswith('.pdf')]
    texts = [extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]

    script_directory = os.path.dirname(os.path.abspath(__file__))
    pdf_testDirectory = os.path.join(script_directory, "..", pdf_testPath)
    pdf_directory = os.path.join(script_directory, "..", pdf_directory)

    client.delete_collection(name="pdf_collection")
