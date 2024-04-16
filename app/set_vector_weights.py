import re
from nltk.corpus import stopwords
from app.pdf_info_extraction import get_pdf_details


def vector_weights(references_list, word_weight=1.2):
    references_text = ''.join(references_list)

    # Define stop words
    stop_words = set(word.lower() for word in stopwords.words('english'))

    # Collect unique words
    all_unique_words = set()
    word_regex = r'\b[A-Za-z]+\b'
    words = re.findall(word_regex, references_text)
    all_unique_words.update([word.lower() for word in words if word.lower() not in stop_words and len(word) > 1])

    # Assign custom weight to each word
    custom_weights = {word: word_weight for word in all_unique_words}

    return custom_weights
