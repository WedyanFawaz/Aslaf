from langchain_community.vectorstores import Chroma
from chromadb_script import (CHROMA_PATH, embeddings)
from langchain.schema import Document

class Retriever():
    def __init__(self):
       self.vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    def get_context(self, query:str) -> list[Document]:
        retriever = self.vectordb.as_retriever(search_kwargs={'k':2})
        docs = retriever.invoke(query)
        return docs
