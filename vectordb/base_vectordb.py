class BaseVectorDB:
    def store(self, texts, embeddings, metadatas):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def similarity_search(self, query_embedding, top_k=5):
        raise NotImplementedError("This method should be overridden by subclasses.")

