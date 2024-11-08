from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os
import shutil
from langchain.embeddings import SentenceTransformerEmbeddings
import re

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


def split_docs_into_sentences(docs: list[Document]):
    """
    Takes documents and splits them into sentences based on full stops.
    Each sentence is treated as a separate document, preserving original metadata.
    """
    split_docs = []


    sentence_splitter_pattern = r'(?<=\.|\؟|\!|\؟)[\s]*'  # all full stops and question marks

    for doc in docs:
        doc_text = doc.page_content
        
        sentences = re.split(sentence_splitter_pattern, doc_text)
        
        for sentence in sentences:
            sentence = sentence.strip()  # Remove leading/trailing spaces
            
            if sentence: 
                new_doc = Document(
                    page_content=sentence,
                    metadata=doc.metadata
                )
                split_docs.append(new_doc)
    
    return split_docs


def save_to_chroma(chunks: list[Document]):
    """Save documents to Chroma vector database."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Remove existing directory if it exists

    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()

def generate_vector_db():
    """Generate the vector database from loaded documents."""
    docs = load_docs()
    chunks = split_docs_into_sentences(docs)
    save_to_chroma(chunks)

# Run the script to generate the vector database
if __name__ == "__main__":
    generate_vector_db()