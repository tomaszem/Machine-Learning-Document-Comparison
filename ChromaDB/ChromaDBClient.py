import os
import fitz  # PyMuPDF
import chromadb
import json
from app import pdf_info_extraction
from app import clustering_operations
from app import set_vector_weights

client = chromadb.HttpClient(host='chromaDB', port=8000)

collection = client.get_or_create_collection(name="pdf_collection")

pdf_testPath = None
texts = None


def update_collection_chromadb(pdf_directory):
    pdf_details = pdf_info_extraction.extract_details(pdf_directory)
    if pdf_details is None:
        return 'No PDF details found'
    references_list = [details['references'] for details in pdf_details.values()]
    custom_weights = set_vector_weights.vector_weights(references_list)
    filenames, reduced_vectors_3d, initial_clusters, reduced_vectors_2d, final_clusters, custom_vectors = clustering_operations.perform_clustering(custom_weights)

    for i, filename in enumerate(os.listdir(pdf_directory)):
        if filename.endswith(".pdf"):
            metadata = pdf_details.get(filename)
            if metadata is None:
                print(f"No metadata found for file: {filename}")
                continue  # Skip this iteration if metadata is None

            # Ensure embeddings are a list of lists of integers or floats
            embeddings_list = [list(map(float, sublist)) for sublist in [custom_vectors[i]]]
            reduced_vectors_3d_list = [reduced_vectors_3d[i].tolist()]
            reduced_vectors_2d_list = [reduced_vectors_2d[i].tolist()]
            initial_clusters_list = [int(initial_clusters[i])]
            final_clusters_list = [int(final_clusters[i])]

            collection.upsert(
                documents=[filename],
                embeddings=embeddings_list,
                metadatas=[{
                    "document": filename,
                    "title": metadata["title"],
                    "authors": metadata["authors"],
                    "abstract": metadata["abstract"],
                    'locations': metadata['locations'],
                    "references": metadata["references"],
                    "reduced_vectors_3d": json.dumps(reduced_vectors_3d_list),
                    "reduce_vectors_2d": json.dumps(reduced_vectors_2d_list),
                    "initial_clusters": json.dumps(initial_clusters_list),
                    "final_clusters": json.dumps(final_clusters_list),
                }],
                ids=[filename]
            )


def extract_text_from_pdf(pdf_directory):
    text = ''
    doc = fitz.open(pdf_directory)
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
    collection = client.get_collection(name="pdf_collection")
    collection = collection.get(include=["metadatas"])
    return collection


# if __name__ == "__main__":
#     pdf_directory = "documents/pdf"
#     pdf_testPath = "../documents/pdf"
#
#     pdf_files = [os.path.join(pdf_testPath, f) for f in os.listdir(pdf_testPath) if f.endswith('.pdf')]
#     texts = [extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]
#
#     script_directory = os.path.dirname(os.path.abspath(__file__))
#     # pdf_testDirectory = os.path.join(script_directory, "..", pdf_testPath)
#     pdf_directory = os.path.join(script_directory, "..", pdf_directory)

# update_collection_chromadb(pdf_testPath)
# client.delete_collection(name="pdf_collection")
