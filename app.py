# app.py
import streamlit as st
from rag_pipeline import answer_query, process_document

st.set_page_config(page_title="FAQ Chatbot", layout="wide")

st.title("Context-Aware FAQ Chatbot with RAG")
st.write("Upload a document, then ask questions!")

# --- Sidebar controls ---
st.sidebar.header(" Model Settings")

temperature = st.sidebar.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
max_length = st.sidebar.slider("Max Output Tokens", 50, 500, 200, 50)
top_k = st.sidebar.slider("Top-k Retrieved Chunks", 1, 10, 3, 1)

# --- File upload ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing document..."):
        process_document(uploaded_file)  # You need to implement this in rag_pipeline
    st.success("Document processed! You can now ask questions.")

    query = st.text_input("Ask a question:")
    if query:
        with st.spinner("Generating answer..."):
            answer, context = answer_query(
                query, top_k=top_k, temperature=temperature, max_length=max_length
            )

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            for c in context:
                st.markdown(f"- {c}")
else:
    st.info("Please upload a PDF to get started.")
