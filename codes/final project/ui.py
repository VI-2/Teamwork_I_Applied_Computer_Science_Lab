import streamlit as st

from backend import DEFAULT_CONFIG, RAGSystem


def apply_blue_theme():
    st.markdown(
        """
        <style>
            .stApp {
                background: radial-gradient(circle at top left, #0a2540 0%, #071324 60%, #050b14 100%);
                color: white;
            }
            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0b3d91 0%, #08254e 60%, #06152a 100%);
                color: white;
            }
            .blue-card {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 18px;
                padding: 18px;
            }
            .title {
                font-size: 34px;
                font-weight: 800;
                letter-spacing: 0.4px;
                margin-bottom: 0px;
            }
            .subtitle {
                opacity: 0.85;
                margin-top: 6px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_system():
    if "system" not in st.session_state:
        config = DEFAULT_CONFIG.copy()
        st.session_state.system = RAGSystem(config)


def main():
    st.set_page_config(page_title="Local RAG", layout="wide")
    apply_blue_theme()
    init_system()

    system = st.session_state.system
    config = system.config

    st.markdown('<div class="title">Local RAG with Ollama</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Text RAG plus image and video semantic search using CLIP.</div>', unsafe_allow_html=True)

    st.sidebar.header("Settings")
    config["llm_model"] = st.sidebar.text_input("LLM model", config["llm_model"])
    config["embedding_model"] = st.sidebar.text_input("Text embedding model", config["embedding_model"])
    config["top_k_retrieval"] = st.sidebar.slider("Top K retrieval", 1, 15, config["top_k_retrieval"])
    config["chunk_size"] = st.sidebar.number_input("Chunk size", 100, 2000, config["chunk_size"], step=50)
    config["chunk_overlap"] = st.sidebar.number_input("Chunk overlap", 0, 500, config["chunk_overlap"], step=10)

    st.sidebar.divider()
    if st.sidebar.button("Clear everything"):
        system.clear_all()
        st.sidebar.success("Cleared")

    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Text chunks", system.text_db.count())
    with col2:
        st.metric("Media items", system.media_db.count())
    with col3:
        st.metric("Top K", config["top_k_retrieval"])

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.markdown('<div class="blue-card">', unsafe_allow_html=True)
        st.subheader("Upload and ingest files")

        uploaded = st.file_uploader(
            "Upload PDF, TXT/MD, images, or videos",
            type=["pdf", "txt", "md", "png", "jpg", "jpeg", "mp4", "mov", "mkv"],
            accept_multiple_files=True,
        )

        if uploaded and st.button("Save and ingest uploads"):
            for f in uploaded:
                path = system.save_uploaded_file_to_data(f)
                result = system.ingest_file(path)
                st.write(f"{f.name} -> {result['kind']} | items added: {result['chunks_added']}")

        if st.button("Ingest everything in data folder"):
            res = system.ingest_data_folder()
            st.write(res)

        st.subheader("Ingest links")
        links_raw = st.text_area("Paste links (one per line)", height=120)
        if st.button("Ingest links now"):
            links = [x.strip() for x in links_raw.split("\n") if x.strip()]
            total = 0
            for link in links:
                total += system.ingest_text_source(link)
            st.write("Text chunks added from links:", total)

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="blue-card">', unsafe_allow_html=True)
        st.subheader("Ask a question")

        q = st.text_input("Question")
        show_chunks = st.checkbox("Show retrieved text chunks", value=False)
        show_media = st.checkbox("Show related media results", value=True)

        if st.button("Run query"):
            out = system.answer(q)

            st.markdown("Answer")
            st.write(out["answer"])

            st.markdown("Sources")
            st.write(out["sources"] if out["sources"] else "No sources")

            if show_chunks:
                st.markdown("Retrieved text chunks")
                for i, ch in enumerate(out["retrieved_chunks"], start=1):
                    src = ch.get("metadata", {}).get("source", "unknown")
                    dist = ch.get("distance", None)
                    with st.expander(f"Chunk {i} | {src} | distance={dist}"):
                        st.write(ch.get("text", ""))

            if show_media:
                st.markdown("Related media results")
                media = out.get("retrieved_media", [])
                if not media:
                    st.write("No media results")
                else:
                    for i, m in enumerate(media, start=1):
                        meta = m.get("metadata", {})
                        st.write(
                            f"{i}. {meta.get('type')} | source={meta.get('source')} | distance={m.get('distance')}"
                        )
                        st.write(m.get("document"))

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    st.subheader("Manage sources")
    sources = system.list_all_sources()
    all_sources = sorted(list(set(sources["text_sources"] + sources["media_sources"])))

    if not all_sources:
        st.write("No sources stored yet.")
        return

    selected = st.selectbox("Select a source to delete", all_sources)
    if st.button("Delete selected source"):
        res = system.delete_source_everywhere(selected)
        st.write(res)
        st.rerun()


if __name__ == "__main__":
    main()
