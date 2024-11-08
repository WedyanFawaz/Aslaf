from langchain_community.document_loaders import DirectoryLoader
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
import os
import shutil
from langchain.embeddings import SentenceTransformerEmbeddings
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter # splitting text into chunks

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

def split_fixed(docs: list[Document]):
    """
    Takes documents and split them into chunks
    """
    # define splitter to chunk docs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 0,
        length_function = len,
        add_start_index = True
    )

    chunks = []

    for doc in docs:

        if len(doc.page_content.split()) > 1000:
            split_chunks = text_splitter.split_documents([doc]) # split each document into chunks

            for chunk in split_chunks:
                chunk.metadata = doc.metadata  # retain original doc metadata
            chunks.extend(split_chunks)
        else:
            chunks.append(doc)

    return chunks

# def save_to_chroma(chunks: list[Document]):
#     """Save documents to Chroma vector database."""
#     if os.path.exists(CHROMA_PATH):
#         shutil.rmtree(CHROMA_PATH)  # Remove existing directory if it exists

#     db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
#     db.persist()


def save_to_chroma(docs: list[Document], batch_size: int = 200):
    """Save documents to Chroma vector database in batches."""
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)  # Remove existing directory if it exists

    # Batch the docs before inserting into the Chroma DB
    batched_docs = [docs[i:i + batch_size] for i in range(0, len(docs), batch_size)]

    for batch in batched_docs:
        # Inserting embeddings directly into Chroma
        db = Chroma.from_documents(batch, embeddings, persist_directory=CHROMA_PATH)
        db.persist()

def generate_vector_db():
    """Generate the vector database from loaded documents."""
    docs = load_docs()
    chunks = split_fixed(docs)
    save_to_chroma(chunks)

# Run the script to generate the vector database
if __name__ == "__main__":
    generate_vector_db()