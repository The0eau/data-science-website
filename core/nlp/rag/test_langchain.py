import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class LocalRAGOrchestrator:
    def __init__(self, docs):
        print("--- Initializing Modern LCEL RAG ---")
        # 1. Setup Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # 2. Setup Vector Store
        self.vector_store = FAISS.from_texts(docs, self.embeddings)
        self.retriever = self.vector_store.as_retriever()
        # 3. Setup Local LLM
        self.llm = ChatOllama(model="llama3", temperature=0)

    def ask(self, query):
        # 4. Define Prompt
        template = """Answer the question based ONLY on the following context:
        {context}
        
        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)

        # 5. Build the Chain using LCEL (No 'langchain.chains' needed!)
        # This pipes: Context -> Prompt -> LLM -> String Output
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return chain.invoke(query)

if __name__ == "__main__":
    my_data = [
        "Theau's favorite programming language is Python.",
        "Theau built a local RAG system on a MacBook Air in 2026.",
        "The secret code for the server is 12345."
    ]
    
    try:
        orchestrator = LocalRAGOrchestrator(my_data)
        question = "What is the secret code and on which computer was the RAG built?"
        
        print(f"\nQuestion: {question}")
        response = orchestrator.ask(question)
        print(f"\nAI Response: {response}")
        
    except Exception as e:
        print(f"\n❌ Final Error: {e}")