import os
import re
import string
import PyPDF4
import fitz  # PyMuPDF
from pdfminer.high_level import extract_text
from pdfminer.pdfparser import PDFSyntaxError


def sanitize(filename):
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    return ''.join([c for c in filename if c in valid_chars])


def sanitize_authors(authors):
    return ''.join([c for c in authors if not c.isdigit()])


def metadata(filename):
    try:
        with open(filename, 'rb') as file:
            reader = PyPDF4.PdfFileReader(file)
            docinfo = reader.getDocumentInfo()
            return docinfo.title if docinfo else ""
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


def pdf_title_authors(filename):
    pdf_document = fitz.open(filename)
    text = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    title = metadata(filename)
    authors = metadata(filename)
    if valid_title(title):
        match = re.search(authors, text)
        if match:
            match_line = text[match.start():text.find('\n', match.start())].strip()
            match_line = re.sub(r'[,*]', '', match_line)
            match_line = re.sub(r'\b\w\b', '', match_line)
            match_line = match_line.replace('  ', ', ')
            authors = match_line
        return title, authors
    title, authors = text_title(filename)
    if valid_title(title):
        lines = pdf_text(filename).strip().split('\n')
        if authors.endswith(','):
            next_line_index = lines.index(authors)
            authors += " " + lines[next_line_index + 1].strip() if next_line_index + 1 < len(lines) else ""
        return title, authors
    return os.path.basename(os.path.splitext(filename)[0]), ""


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
    start_page = max(0, num_pages - 3)
    for page_num in range(start_page, num_pages):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    pdf_document.close()
    references_match = (re.search(r'\bReferences\b\s*(.*)', text, re.IGNORECASE | re.DOTALL))
    if references_match:
        references = references_match.group(1).strip()
    return references


def extract_details(directory):
    pdf_details = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            abstract = extract_abstract(file_path)
            title, authors = pdf_title_authors(file_path)
            references = extract_references(file_path)
            pdf_details[filename] = {"abstract": abstract, "title": title, "authors": authors, "references": references}
    return pdf_details


directory_path = "../documents"
pdf_details = extract_details(directory_path)

for filename, info in pdf_details.items():
    abstract = info['abstract']
    references = info['references']
    title = info["title"]
    authors = sanitize_authors(info["authors"])

    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\nNext Document:\n")
    print(f"Filename: {filename}\nTitle: {title}\nAuthors: {authors}\nAbstract:\n{abstract}\nReferences:\n{references}\n")
