from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from transformers import BertTokenizer, BertModel
import numpy as np

CONNECTION_STRING = "postgresql+psycopg2://vectordb:vectordbv@localhost:5432/vectordb"
COLLECTION_NAME = "sales_documents"

# Load the pre-trained BERT model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

embeddings = HuggingFaceEmbeddings()

try:
    # Create PGVector instance
    db = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings.embed_documents
    )

    def query_similar_documents(query_text, top_k=5):
        # Preprocess the query text
        inputs = tokenizer(query_text, return_tensors='pt')
        outputs = model(**inputs)
        query_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy().flatten().tolist()  # Convert to flattened list of floats

        # Print the query embedding for debugging
       # print("Query Embedding:", query_embedding)

        # Retrieve similar documents based on the query embedding
        results = db.similarity_search_with_score_by_vector(query_embedding, k=top_k)
        
        # Print the results for debugging
       # print("Results:", results)

        # Calculate similarity scores and add them to the results
        similarity_scores = []
        for result, score in results:
            print("Processing result:", result)
            # Ensure that the result has the metadata and embedding stored correctly
            if 'embedding' in result.metadata:
                doc_embedding = result.metadata['embedding']
                if isinstance(doc_embedding, str):  # If the embedding is stored as a string, convert it back to a list of floats
                    doc_embedding = list(map(float, doc_embedding.strip('[]').split(',')))
                similarity_score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
                similarity_scores.append((similarity_score, result.document, result.metadata.get("source", "Source URL not found")))

        # Sort the similarity scores in descending order based on the similarity score
        similarity_scores.sort(reverse=True, key=lambda x: x[0])

        return similarity_scores

    # Example query to find similar documents
    query_text = "Red Hat OpenShift AI content cheatsheet?"
    similar_docs = query_similar_documents(query_text)
    print("Similar Docs: ", similar_docs)

    # Print the top results
    for score, content, source in similar_docs:
        print("Similarity Score:", score)
        print("Document Content:", content)
        print("Source URL:", source)
        print("\n---\n")

finally:
    # Ensure proper cleanup to avoid the AttributeError
    if 'db' in locals():
        del db  # Delete the object manually to avoid __del__ call
