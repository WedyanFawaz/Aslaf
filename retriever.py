from langchain_community.vectorstores import Chroma
from chromadb_script import (CHROMA_PATH, embeddings)
from langchain.schema import Document

class Retriever():
    def __init__(self):
       self.vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


    def format_docs(self,docs: list[Document]) -> str:
        context =  "\n\n".join(doc.page_content for doc in docs)
        resources = "\n\n".join(doc.metadata['source'] for doc in docs).replace("data/", "").replace(".txt", "")
        return (context, resources)


    def get_context(self, query: str) -> str:
        retriever = self.vectordb.as_retriever(search_kwargs={'k': 3})

        docs_with_scores = self.vectordb.similarity_search_with_relevance_scores(query, k=3)

        filtered_docs = [
            doc for doc, score in docs_with_scores if score >= 0.735
        ]

        context = self.format_docs(filtered_docs)

        return context

