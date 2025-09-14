# Conversational_qa_chatbot
app.py
Main entry point of the chatbot application. It provides the interface to interact with the RAG pipeline.

document_ingestion.py
Handles ingestion of documents (e.g., PDFs, text files, or website content). Splits documents into chunks and generates embeddings for storage.

rag_pipeline.py
Implements the RAG pipeline logic:

Encodes queries

Retrieves relevant chunks from the vector store

Passes context to the language model for generating answers

vector_store.py
Manages the vector database (using FAISS or similar). Stores and retrieves embeddings efficiently.

Features:
Ingest and process documents into vector embeddings

Store embeddings in a FAISS vector store

Retrieve top-k relevant chunks for queries

Use an LLM to generate answers augmented with retrieved context

Simple application interface
