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


    def get_context(self, query:str) -> str:
        retriever = self.vectordb.as_retriever(search_kwargs={'k':2})
        docs = retriever.invoke(query)
        context = self.format_docs(docs)
        return context


if __name__ == "__main__":
    ret = Retriever()
    _, res = ret.get_context("be")
    print(res)