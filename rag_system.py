import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # Changed from Chroma
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader


class StudyRAGSystem:
    def __init__(self):
        # Use CPU-friendly embeddings
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu"}  # Force CPU usage
        )
        
        # Initialize lightweight LLM
        self.llm = pipeline(
            "text-generation",
            model="mistralai/Mistral-7B-Instruct-v0.1",
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        os.makedirs("data", exist_ok=True)
    
    def index_documents(self, file_paths):
        documents = []
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.docx'):
                loader = Docx2txtLoader(file_path)
            else:
                continue
            documents.extend(loader.load())
        
        split_docs = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(  # Using FAISS
            split_docs,
            self.embedding_model
        )
    
    def retrieve_relevant_docs(self, query):
        if self.vector_store is None:
            return []
        return self.vector_store.similarity_search(query, k=4)
    
    def generate_student_response(self, query):
        docs = self.retrieve_relevant_docs(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""You are a helpful study assistant. Based on:
        {context}
        
        Provide detailed notes about: {query}"""
        
        return self.llm(prompt, max_length=2000)[0]['generated_text']
    
    def generate_faculty_response(self, query):
        docs = self.retrieve_relevant_docs(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Create assessment materials about: {query}
        Include:
        - 5 MCQs (with answers)
        - 3 True/False (with answers)
        - 2 Short Answer (with sample answers)
        - 1 Essay (with rubric)
        
        Context: {context}"""
        
        return self.llm(prompt, max_length=3000)[0]['generated_text']
