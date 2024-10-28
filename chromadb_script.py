from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os
import shutil
from langchain.embeddings import SentenceTransformerEmbeddings

# Constants
CHROMA_PATH = "vectordb"
DATA_PATH = "data/الخصائص"

# Embedding initialization
embeddings = SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-base")

def load_docs() -> list[Document]:
    """Load documents from the specified directory."""
    loader = DirectoryLoader(DATA_PATH, glob="**/*.txt")
    docs = loader.load()
    return docs

def save_to_chroma(docs: list[Document]):
    """Save documents to Chroma vector database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Remove existing directory if it exists

    db = Chroma.from_documents(docs, embeddings, persist_directory=CHROMA_PATH)
    db.persist()

def generate_vector_db():
    """Generate the vector database from loaded documents."""
    docs = load_docs()
    save_to_chroma(docs)

# Run the script to generate the vector database
if __name__ == "__main__":
    generate_vector_db()
