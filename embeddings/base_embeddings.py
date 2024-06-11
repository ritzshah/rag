class BaseEmbeddings:
    def embed(self, texts):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def embed_query(self, text):
        raise NotImplementedError("This method should be overridden by subclasses.")

