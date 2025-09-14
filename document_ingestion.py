# document_ingestion.py
import fitz  # PyMuPDF
import argparse
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text

def chunk_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and chunk a PDF document")
    parser.add_argument("file_path", type=str, help="Path to the PDF file")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="Chunk overlap")
    args = parser.parse_args()

    raw_text = load_pdf(args.file_path)
    chunks = chunk_text(raw_text, args.chunk_size, args.chunk_overlap)

    print(f"Loaded {len(chunks)} chunks from {args.file_path}")
    print(chunks[:2])
