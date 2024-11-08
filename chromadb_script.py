from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os
import shutil
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import NLTKTextSplitter

# Constants
CHROMA_PATH = "vectordb_ibnjenni"
DATA_PATH = "data"

# Embedding initialization
embeddings = SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-base")

def load_docs() -> list[Document]:
    """Load documents from the specified directory."""
    loader = DirectoryLoader(DATA_PATH)
    docs = loader.load()
    return docs

def split_docs(docs: list[Document]):
    """
    Takes documents and split them into chunks
    """
    splitter = NLTKTextSplitter(chunk_size=100, chunk_overlap=20)

    chunks = []
    for doc in docs:
        # Split each document into chunks using the NLTKTextSplitter
        split_chunks = splitter.split_text(doc.page_content)

        # Retain original doc metadata for each chunk
        for chunk in split_chunks:
            chunk.metadata = doc.metadata
        
        # Append the chunks to the chunks list
        chunks.extend(split_chunks)

    return chunks


def save_to_chroma(chunks: list[Document]):
    """Save documents to Chroma vector database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Remove existing directory if it exists

    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()

def generate_vector_db():
    """Generate the vector database from loaded documents."""
    docs = load_docs()
    chunks = split_docs(docs)
    save_to_chroma(chunks)

# Run the script to generate the vector database
if __name__ == "__main__":
    generate_vector_db()
