from loaders import PDFLoader
from splitters import RecursiveSplitter
#from embeddings import HuggingFaceEmbeddings
from vectordb import PGVectorDB
from utils import update_pdf_metadata
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings

CONNECTION_STRING = "postgresql+psycopg://vectordb:vectordbv@localhost:5432/vectordb"
COLLECTION_NAME = "sales_documents"
pdf_folder_path = "../sales"
csv_file_path = Path(pdf_folder_path) / "sales-csv.csv"

# Load the PDF documents
pdf_loader = PDFLoader(pdf_folder_path)
pdf_docs = pdf_loader.load()

# Update the metadata with URLs from the CSV file
update_pdf_metadata(pdf_docs, csv_file_path)

# Split the text documents
text_splitter = RecursiveSplitter()
all_splits = text_splitter.split_documents(pdf_docs)

# Remove null characters from the page content
for doc in all_splits:
    doc.page_content = doc.page_content.replace('\x00', '')

print("Text splitting and null character removal completed.")

# Generate embeddings
embeddings = HuggingFaceEmbeddings()  # Instantiate HuggingFaceEmbeddings class

# Extract text content for embedding
text_contents = [doc.page_content for doc in all_splits if isinstance(doc.page_content, str)]

# Store documents and embeddings in PGVector
db = PGVectorDB(COLLECTION_NAME, CONNECTION_STRING, embeddings)
db.store(text_contents, embeddings, [doc.metadata for doc in all_splits if isinstance(doc.page_content, str)])

print("Documents and embeddings stored in PGVector.")

'''
# Query similar documents
def query_similar_documents(query_text, top_k=5):
    query_embedding = embeddings.embed_query(query_text)
    results = db.similarity_search(query_embedding, top_k)
    
    similarity_scores = []
    for result in results:
        similarity_score = 1 - result[2]  # Assuming distance is returned
        similarity_scores.append((similarity_score, result[0], result[1].get("source", "Source URL not found")))

    similarity_scores.sort(reverse=True)
    return similarity_scores[:top_k]

# Example query
query_text = "App Modernization_TIBCO Alternative.pdf"
similar_docs = query_similar_documents(query_text)

for score, content, source in similar_docs:
    print("Similarity Score:", score)
    print("Document Content:", content)
    print("Source URL:", source)
    print("\n---\n")
'''