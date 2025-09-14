# vector_store.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
import os
from document_ingestion import load_pdf, chunk_text


def build_faiss_index_from_pdf(file_path, index_file="vector_store.pkl"):
    """Build FAISS index directly from a PDF file path."""
    text = load_pdf(file_path)
    chunks = chunk_text(text)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index + chunks + model
    with open(index_file, "wb") as f:
        pickle.dump((index, chunks, model), f)
    print(f"FAISS index built and saved to {index_file} with {len(chunks)} chunks.")


def build_faiss_index_from_text(chunks, index_file="vector_store.pkl"):
    """Build FAISS index from pre-chunked text (used after upload in Streamlit)."""
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    with open(index_file, "wb") as f:
        pickle.dump((index, chunks, model), f)
    print(f"FAISS index built and saved to {index_file} with {len(chunks)} chunks.")


def load_faiss_index(index_file="vector_store.pkl"):
    if not os.path.exists(index_file):
        raise FileNotFoundError(f"No FAISS index found at {index_file}. Please build it first.")
    with open(index_file, "rb") as f:
        return pickle.load(f)  # index, chunks, model


if __name__ == "__main__":
    # Example: build index from a PDF (pass file dynamically)
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index from a PDF")
    parser.add_argument("file_path", type=str, help="Path to PDF file")
    parser.add_argument("--index_file", type=str, default="vector_store.pkl", help="Where to save the FAISS index")
    args = parser.parse_args()

    build_faiss_index_from_pdf(args.file_path, args.index_file)
