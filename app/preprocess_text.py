import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


# Function to download nltk resources if not already downloaded
def download_nltk_resources():
    resources = ["stopwords", "punkt"]
    for resource_name in resources:
        try:
            nltk.data.find(f"tokenizers/{resource_name}")
        except LookupError:
            nltk.download(resource_name)


# Set of stopwords
stop_words = set(stopwords.words('english'))


def text_normalization(text):
    # Call the function to ensure resources are downloaded
    download_nltk_resources()

    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text to handle words properly
    words = word_tokenize(text)
    # Remove all non-alphanumeric characters from each word and filter out stopwords
    filtered_words = [re.sub(r'\W+', '', word) for word in words if word not in stop_words and word != '']
    # Join the filtered words back into a single string
    return ' '.join(filtered_words)
