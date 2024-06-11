from langchain.vectorstores import PGVector
from typing import List

class PGVectorDB:
    def __init__(self, collection_name: str, connection_string: str, embedding_function):
        self.collection_name = collection_name
        self.connection_string = connection_string
        self.embedding_function = embedding_function
        self.db = None

    def store(self, texts: List[str], embeddings, metadatas: List[dict]):
        # Delete existing collection if it exists
        if self.db is not None:
            self.db.delete_collection()
        # Store documents and embeddings in PGVector
        self.db = PGVector.from_texts(
            texts=texts,
            embedding=self.embedding_function,
            metadatas=metadatas,
            collection_name=self.collection_name,
            connection_string=self.connection_string,
            pre_delete_collection=True  # This deletes existing collection and its data, use carefully!
        )

    def query(self, query_text: str, top_k: int = 5):
        query_embedding = self.embedding_function([query_text])[0]
        results = self.db.similarity_search_by_vector(query_embedding, k=top_k)
        return results
