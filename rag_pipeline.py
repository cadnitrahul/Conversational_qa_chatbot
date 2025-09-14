# rag_pipeline.py
import numpy as np
import pickle
import faiss
from transformers import pipeline
from document_ingestion import load_pdf, chunk_text
from vector_store import build_faiss_index_from_text


def process_document(uploaded_file, index_file="vector_store.pkl"):
    """
    Process uploaded PDF: extract text, split into chunks,
    and build FAISS index for retrieval.
    """
    # Read PDF bytes from Streamlit upload
    pdf_bytes = uploaded_file.read()

    # Save temporarily (PyMuPDF needs a path or bytes)
    temp_file = "temp_uploaded.pdf"
    with open(temp_file, "wb") as f:
        f.write(pdf_bytes)

    # Extract and chunk
    text = load_pdf(temp_file)
    chunks = chunk_text(text)

    # Build FAISS index from chunks
    build_faiss_index_from_text(chunks, index_file=index_file)
    return len(chunks)


def retrieve(query, index_file="vector_store.pkl", top_k=3):
    """
    Retrieve top-k relevant chunks from FAISS index for a given query.
    """
    with open(index_file, "rb") as f:
        index, chunks, model = pickle.load(f)

    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return retrieved_chunks


def answer_query(query, index_file="vector_store.pkl", top_k=3, temperature=0.7, max_length=200):
    """
    Generate an answer using retrieved context and a text2text model.
    """
    retrieved_chunks = retrieve(query, index_file, top_k=top_k)
    context = "\n".join(retrieved_chunks)

    qa_model = pipeline(
        "text2text-generation",
        model="google/flan-t5-base"
    )

    prompt = f"Answer the question based on context:\nContext: {context}\n\nQuestion: {query}\nAnswer:"
    answer = qa_model(
        prompt,
        max_length=max_length,
        do_sample=True if temperature > 0 else False,
        temperature=temperature
    )

    return answer[0]['generated_text'], retrieved_chunks
