import os
import fitz  # PyMuPDF
import chromadb
import chromadb.utils.embedding_functions as emb_fn

default_ef = emb_fn.DefaultEmbeddingFunction()
client = chromadb.HttpClient(host='localhost', port=8000)

collection = client.get_or_create_collection(name="pdf_collection", embedding_function=default_ef)

pdf_testPath = None
texts = None


def extract_text_and_images_from_pdf(pdf_path):
    text = ''
    images = []

    doc = fitz.open(pdf_path)

    for page_num in range(doc.page_count):
        page = doc[page_num]

        # Extract text
        text += page.get_text()

        # Extract images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_extension = base_image["ext"]
            image_filename = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_page_{page_num + 1}_image_{img_index + 1}.{image_extension}"
            images.append((image_filename, image_bytes))

    return text, images


def embed_pdf_text_and_images_in_chromadb(pdf_directory):
    for filename in os.listdir(pdf_directory):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, filename)
            text, images = extract_text_and_images_from_pdf(pdf_path)
            # Embed text
            collection.upsert(
                documents=[filename],
                metadatas=[{"filename": filename}],
                ids=[filename]
            )
            # Embed images
            # for image_filename, _ in images:
            #     collection.upsert(
            #         images=[image_filename],
            #         metadatas=[{"filename": filename}],
            #         ids=[image_filename]
            #     )

    print(collection.get(include=['embeddings', 'documents', 'metadatas']))
    print(collection.count())


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
        include=["documents", "distances"],
        n_results=4
    )
    print(results)


if __name__ == "__main__":
    pdf_directory = "documents"
    pdf_testPath = "compareDocuments"

    pdf_files = [os.path.join(pdf_testPath, f) for f in os.listdir(pdf_testPath) if f.endswith('.pdf')]
    texts = [extract_text_from_pdf(pdf_file) for pdf_file in pdf_files]
    perform_query(texts)

    script_directory = os.path.dirname(os.path.abspath(__file__))
    pdf_testDirectory = os.path.join(script_directory, "..", pdf_testPath)
    pdf_directory = os.path.join(script_directory, "..", pdf_directory)

    embed_pdf_text_and_images_in_chromadb(pdf_directory)
