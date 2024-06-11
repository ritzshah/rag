This repo is for RAG specific experiments.

EVerything is run locally on M1 based MacBook.

Environment's being setup :
1) Virtual Environment using python 3.10 is used.
2) pip install following (check requirements.txt) / pip freeze output.
   - pip install pgvector pypdf psycopg langchain sentence-transformers langchain-community langchain-experimental langchain-text-splitters pandas PyMuPDF
3) ollama (ollama pull)
 - llama2:latest              
 - sroecker/merlinite:latest    
 - mxbai-embed-large:latest (#EMBEDDINGS)      
 - sroecker/granite-7b-lab:latest	 	
 - mistral:latest                
4) PGVECTOR DB (Postgres as Vector Database) and pgadmin UI
   To start service run `brew services start postgresql`
5) jupyter lab (Run this from where you have you have this git repo cloned locally)


