import streamlit as st

from rag_backend import (
    VectorDB,
    RAGPipeline,
    RAGEvaluator
)
from sentence_transformers import SentenceTransformer


# =============================================================================
# SETUP
# =============================================================================

st.set_page_config(page_title="Local RAG System", layout="wide")

@st.cache_resource
def load_backend():
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    vector_db = VectorDB(
        persist_directory="./chroma_db",
        embedding_model=embedding_model
    )

    rag_pipeline = RAGPipeline(
        vector_db=vector_db,
        llm_model="llama3.1",
        top_k=5
    )

    evaluator = RAGEvaluator(embedding_model)

    return rag_pipeline, evaluator


rag_pipeline, evaluator = load_backend()


# =============================================================================
# UI
# =============================================================================

st.title("Local Retrieval-Augmented Generation")

question = st.text_input("Ask a question")

if st.button("Run RAG") and question:
    response = rag_pipeline.query(question)

    st.subheader("Answer")
    st.write(response["answer"])

    st.subheader("Sources")
    for src in response["sources"]:
        st.write("-", src)

    chunks = [c["chunk"] for c in response["retrieved_chunks"]]
    scores = evaluator.evaluate(question, response["answer"], chunks)

    st.subheader("Evaluation")
    st.write(scores)
