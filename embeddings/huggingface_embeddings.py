from sentence_transformers import SentenceTransformer

class HuggingFaceEmbeddings:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]):
        return self.model.encode(texts, convert_to_tensor=True)

    def embed_query(self, query: str):
        return self.model.encode([query], convert_to_tensor=True)[0]
