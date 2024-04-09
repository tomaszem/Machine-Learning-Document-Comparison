"""
Initialize a list comprehension to iterate over each item in pdf_details.
pdf_details is expected to be a list of dictionaries, where each dictionary
contains details extracted from a PDF.
"""


def pdf_info_to_json(pdf_details):
    json_data = [
        {
            "filename": details['filename'],
            "title": details['title'],
            "authors": details['authors'],
            "abstract": details['abstract'],
            "references": details['references']
        }
        for details in pdf_details
    ]

    return json_data
