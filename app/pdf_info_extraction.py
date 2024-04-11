import os
import fitz  # PyMuPDF
import re
import PyPDF4
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError
import string
import spacy
from app.config.constants import PDF_PATH


def sanitize(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join([c for c in filename if c in valid_chars])


def sanitize_authors(authors):
    authors = authors.replace('\n', '')
    return ''.join([c for c in authors if not c.isdigit()])


def remove_email_addresses(string):
    r = re.compile(r'[\w\.-]+@[\w\.-]+')
    return r.sub('', string)


def metadata(filename):
    try:
        with open(filename, 'rb') as file:
            reader = PyPDF4.PdfFileReader(file)
            docinfo = reader.getDocumentInfo()
            return docinfo if docinfo else ""
    except Exception:
        return ""


def copyright_line(line):
    return re.search(r'technical\s+report|proceedings|preprint|to\s+appear|submission', line.lower())


def empty_str(s):
    return len(s.strip()) == 0


def pdf_text(filename):
    try:
        text = extract_text(filename)
        return text
    except (PDFSyntaxError, Exception):
        return ""


def find_persons_locations(filename, title):
    nlp = spacy.load("en_core_web_sm")
    text = extract_text(filename)
    lines = text.split('\n')

    # Take lines until "Abstract" is found
    abstract_found = False
    first_lines = []
    for line in lines:
        if abstract_found:
            break
        if "abstract" in line.lower() or "a b s t r a c t" in line.lower():
            abstract_found = True
        if line.strip():  # Check if line is non-empty
            first_lines.append(line.strip())  # Remove leading/trailing whitespace

    sanitized_text = '\n'.join(first_lines)
    doc = nlp(sanitized_text)
    persons = []
    locations = []

    for ent in doc.ents:
        if ent.label_ == "GPE":
            clean_text = ent.text.strip().replace(".", "")
            if clean_text.replace(" ", "").isalpha():
                locations.append(clean_text)
        if ent.label_ == "PERSON" and "ORG" not in ent.text:
            persons.append(ent.text.strip())  # Remove leading/trailing whitespace

    persons = sanitize_authors(', '.join(persons))
    persons = remove_email_addresses(persons)
    locations = ', '.join(list(set(locations)))  # Convert list of locations to a string

    # Remove parts of title from persons
    title_parts = title.split()
    for part in title_parts:
        if part in persons:
            persons = persons.replace(part, '')

    return persons, locations


def title_start(lines):
    for i, line in enumerate(lines):
        if not empty_str(line) and not copyright_line(line):
            return i
    return 0


def title_end(lines, start, max_lines=2):
    for i, line in enumerate(lines[start + 1:], start + 1):
        if empty_str(line):
            return i
    return len(lines)


def text_title(filename):
    lines = pdf_text(filename).strip().split('\n')

    i = title_start(lines)
    j = title_end(lines, i)

    title = ' '.join(line.strip() for line in lines[i:j])

    next_line_index = j
    while next_line_index < len(lines) and empty_str(lines[next_line_index]):
        next_line_index += 1
    authors = lines[next_line_index].strip() if next_line_index < len(lines) else ""
    return title, authors


def valid_title(title):
    return not empty_str(title)


def pdf_title(filename):
    pdf_document = fitz.open(filename)

    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()

    title = metadata(filename).get('/Title', "")

    if valid_title(title):
        return title  # Return only title

    title, _ = text_title(filename)
    if valid_title(title):
        return title

    return os.path.basename(os.path.splitext(filename)[0])


def extract_abstract(pdf_path):
    pdf_document = fitz.open(pdf_path)

    abstract = None

    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()

        abstract_match = (
                re.search(r'(?s)((?<=\bAbstract\b).*?(?=\b(?:\d*\s*)?Introduction\b))', text, re.IGNORECASE) or
                re.search(r'(?s)((?<=\bA b s t r a c t\b).*?(?=\b(?:\d*\s*)?Introduction\b))', text, re.IGNORECASE))
        if abstract_match:
            abstract = abstract_match.group(1).strip()
            break

    pdf_document.close()

    return abstract


def extract_references(pdf_path):
    pdf_document = fitz.open(pdf_path)
    references = None
    num_pages = len(pdf_document)
    text = ""
    references_found = False
    appendix_found = False

    for page_num in range(num_pages - 1, -1, -1):
        page = pdf_document.load_page(page_num)
        text = page.get_text() + text
        if re.search(r'\bReferences\b', text, re.IGNORECASE) or re.search(r'\bReference\b', text, re.IGNORECASE):
            references_found = True
            break
        elif re.search(r'\bAppendix\b', text, re.IGNORECASE):
            appendix_found = True

    if references_found:
        # Check for "Appendix" after the references
        if appendix_found:
            appendix_index = re.search(r'\bAppendix\b', text, re.IGNORECASE)
            text = text[:appendix_index.start()]

        references_match = re.search(r'\bReference(?:s)?\b\s*(.*)', text, re.IGNORECASE | re.DOTALL)
        if references_match:
            references = references_match.group(1).strip()

    pdf_document.close()
    return references


def extract_details(directory):
    pdf_details = {}

    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)

            abstract = extract_abstract(file_path) or "Nothing was found"
            title = pdf_title(file_path) or "Nothing was found"
            references = extract_references(file_path) or "Nothing was found"
            authors, locations = find_persons_locations(file_path, title) or ("Nothing was found", "Nothing was found")

            pdf_details[filename] = {"abstract": abstract, "title": title, "authors": authors, "locations": locations,
                                     "references": references}

    return pdf_details


def get_pdf_details():
    directory_path = PDF_PATH
    pdf_details = extract_details(directory_path)
    extracted_details = []
    for filename, info in pdf_details.items():
        details = {
            'filename': filename,
            'title': info['title'],
            'authors': info['authors'],
            'locations': info['locations'],
            'abstract': info['abstract'],
            'references': info['references']
        }
        extracted_details.append(details)
    return extracted_details


# For testing purposes only
"""
directory_path = "../documents/pdf"
pdf_details = extract_details(directory_path)

for filename, info in pdf_details.items():
    abstract = info["abstract"]
    references = info["references"]
    title = info["title"]
    authors = info["authors"]
    locations = info["locations"]

    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nNext Document:\n")
    print(
        f"Filename: {filename}\nTitle: {title}\nAuthors: {authors}\nLocations: {locations}\nAbstract:\n{abstract}\nReferences:\n{references}\n")
"""
