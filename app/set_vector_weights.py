import re
from nltk.corpus import stopwords
from app.pdf_info_extraction import get_pdf_details

stop_words = set(stopwords.words('english'))


def vector_weights():
    custom_weights = {}

    all_pdf_details = get_pdf_details()
    custom_references = [detail['references'] for detail in all_pdf_details]

    references_text = ''.join([ref for sublist in custom_references if sublist for ref in sublist if ref])

    stop_words = set(word.lower() for word in stopwords.words('english'))
    all_unique_words = set()

    word_regex = r'\b[A-Za-z]+\b'

    words = re.findall(word_regex, references_text)
    all_unique_words.update([word.lower() for word in words if word.lower() not in stop_words and len(word) > 1])

    custom_weights = {word: 1.3 for word in all_unique_words}

    # TODO return custom_weights
    # custom_vectors = custom_vectorization(texts, custom_weights)


vector_weights()
