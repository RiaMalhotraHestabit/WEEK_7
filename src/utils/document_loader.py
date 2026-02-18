import os
from pypdf import PdfReader


def load_pdf(file_path: str):
    documents = []

    reader = PdfReader(file_path)
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            documents.append({
                "text": text,
                "metadata": {
                    "source": os.path.basename(file_path),
                    "page": page_number + 1
                }
            })

    return documents


def load_documents_from_folder(folder_path: str):
    all_docs = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)

        if file.endswith(".pdf"):
            docs = load_pdf(file_path)
            all_docs.extend(docs)

    return all_docs
