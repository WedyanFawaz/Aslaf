from langchain_community.vectorstores import Chroma
from chromadb_script import (CHROMA_PATH, embeddings)
from langchain.schema import Document

class Retriever():
    def __init__(self):
       self.vectordb = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)


    def format_docs(self, docs: list[Document]) -> str:
        context = "\n\n".join(doc.page_content for doc, score in docs if score >= 0.71)
        resources = "\n\n".join(
            doc.metadata['source'] for doc, score in docs if score > 0.73
        ).replace("data/", "").replace(".txt", "")
        return context, resources

    def get_context(self, query: str) -> str:
        # Perform similarity search with relevance scores
        docs_with_scores = self.vectordb.similarity_search_with_relevance_scores(query, k=3)

        # Filter documents with a score of at least 0.71
        # filtered_docs = [
        #     doc for doc, score in docs_with_scores if score >= 0.71
        # ]

        # Generate context and filtered sources
        context, resources = self.format_docs(docs_with_scores)

        return context, resources
