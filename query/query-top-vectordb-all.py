from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector

CONNECTION_STRING = "postgresql+psycopg2://vectordb:vectordbv@localhost:5432/vectordb"
COLLECTION_NAME = "sales_documents"

embeddings = HuggingFaceEmbeddings()

try:
    # Create PGVector instance
    db = PGVector(
        collection_name=COLLECTION_NAME,
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings.embed_documents
    )

    def query_similar_documents(query_text, top_k=5):
        # Generate the query embedding using the correct method
        query_embedding = embeddings.embed_query(query_text)
        print("Query Embeddings: ", query_embedding)
        # Retrieve similar documents based on the query embedding
        results = db.similarity_search_by_vector(query_embedding, k=top_k)
        
        # Calculate similarity scores and add them to the results
        similarity_scores = []
        for result in results:
            # Compute the similarity score between the query and each document
            doc_embedding = embeddings.embed_query(result.page_content)
            # Compute cosine similarity between query and document embeddings
            similarity_score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
            similarity_scores.append((similarity_score, result.page_content, result.metadata.get("source", "Source URL not found")))

        # Sort the similarity scores in descending order based on the similarity score
        similarity_scores.sort(reverse=True)

        return similarity_scores

    # Example query to find similar documents
    query_text = "Red Hat OpenShift AI 101 sales play"
    similar_docs = query_similar_documents(query_text)

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
