import os
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor
from app.preprocess_text import text_normalization

folder_path = 'documents'

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text


# Processing files in a batch
def process_files_batch(files_batch):
    texts = []
    filenames = []
    for file in files_batch:
        file_path = os.path.join(folder_path, file)
        text = extract_text_from_pdf(file_path)
        cleaned_text = text_normalization(text)
        texts.append(cleaned_text)
        filenames.append(file)
    return texts, filenames


# Loading texts from multiple PDF files in batches
def load_texts_from_pdfs_batched(batch_size):
    all_texts = []
    all_filenames = []
    files = [file for file in os.listdir(folder_path) if file.endswith('.pdf')]

    for i in range(0, len(files), batch_size):
        files_batch = files[i:i + batch_size]
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_files_batch, [file]) for file in files_batch]
            for future in futures:
                texts, filenames = future.result()
                all_texts.extend(texts)
                all_filenames.extend(filenames)

    return all_texts, all_filenames

# Path to your stopwords YAML file (update this to the correct path)
# stopwords_file_path = 'config/stopwords.yaml'

# The alternative way using custom stopwords set
# def load_stopwords(file_path):
#     with open(file_path, 'r') as file:
#         stopwords = yaml.safe_load(file)
#     return set(stopwords)

# Load the stopwords
# stopwords_set = load_stopwords(stopwords_file_path)
