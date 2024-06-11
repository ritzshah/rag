import os
import fitz  # PyMuPDF
from .base_loader import BaseLoader

class PDFDocument:
    def __init__(self, doc, metadata, text_content):
        self.doc = doc
        self.metadata = metadata
        self.page_content = text_content

    def get_text_content(self):
        return self.page_content

class PDFLoader(BaseLoader):
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def load(self):
        pdf_docs = []
        for filename in os.listdir(self.directory_path):
            if filename.endswith(".pdf"):
                filepath = os.path.join(self.directory_path, filename)
                pdf_doc = self.load_pdf(filepath)
                pdf_docs.append(pdf_doc)
        return pdf_docs

    def load_pdf(self, filepath):
        doc = fitz.open(filepath)
        metadata = doc.metadata
        metadata['source'] = filepath
        text_content = ""
        for page in doc:
            text_content += page.get_text()
        return PDFDocument(doc, metadata, text_content)

