from langchain.text_splitter import RecursiveCharacterTextSplitter
from .base_splitter import BaseSplitter

class RecursiveSplitter(BaseSplitter):
    def __init__(self, chunk_size=1024, chunk_overlap=40):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    def split_documents(self, documents):
        return self.splitter.split_documents(documents)

