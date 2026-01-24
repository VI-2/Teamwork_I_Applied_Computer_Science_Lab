import streamlit as st
import time
from backend import DEFAULT_CONFIG, RAGSystem, RAG_Evaluator



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
                border-right: 1px solid rgba(255,255,255,0.10);
            }

            .title {
                font-size: 30px;
                font-weight: 900;
                letter-spacing: 0.4px;
                margin-bottom: 6px;
            }

            .subtitle {
                opacity: 0.85;
                margin-top: 0px;
                margin-bottom: 14px;
            }

            div[data-testid="stMetric"] {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.12);
                padding: 14px;
                border-radius: 14px;
            }

            div[data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 18px;
            }

            .stButton button {
                border-radius: 14px;
                border: 1px solid rgba(255,255,255,0.18);
                background: rgba(59,130,246,0.22);
            }

            .stButton button:hover {
                border: 1px solid rgba(255,255,255,0.28);
                background: rgba(59,130,246,0.35);
                transition: 0.15s;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state():
    if "system" not in st.session_state:
        config = DEFAULT_CONFIG.copy()
        st.session_state.system = RAGSystem(config)

    if "evaluator" not in st.session_state:
        st.session_state.evaluator = RAG_Evaluator()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm Chatty. Upload docs and ask me stuff."}
        ]

    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None

    if "last_evaluation" not in st.session_state:
        st.session_state.last_evaluation = None

    if "ui_show_sources" not in st.session_state:
        st.session_state.ui_show_sources = True

    if "ui_show_chunks" not in st.session_state:
        st.session_state.ui_show_chunks = False

    if "ui_show_media" not in st.session_state:
        st.session_state.ui_show_media = True


def sidebar_settings(system: RAGSystem):
    st.sidebar.markdown("## Settings")

    system.config["llm_model"] = st.sidebar.text_input("LLM model", system.config["llm_model"])
    system.config["embedding_model"] = st.sidebar.text_input(
        "Text embedding model", system.config["embedding_model"]
    )

    system.config["top_k_retrieval"] = st.sidebar.slider(
        "Top K retrieval", 1, 15, int(system.config["top_k_retrieval"])
    )

    system.config["chunk_size"] = st.sidebar.number_input(
        "Chunk size", 100, 2000, int(system.config["chunk_size"]), step=50
    )

    system.config["chunk_overlap"] = st.sidebar.number_input(
        "Chunk overlap", 0, 500, int(system.config["chunk_overlap"]), step=10
    )

    st.sidebar.divider()

    colA, colB = st.sidebar.columns(2)

    with colA:
        if st.button("Clear chat"):
            st.session_state.messages = [
                {"role": "assistant", "content": "Chat cleared. Database still saved."}
            ]
            st.session_state.last_answer = None
            st.session_state.last_evaluation = None 
            st.rerun()

    with colB:
        if st.button("Clear DB"):
            res = system.clear_all()
            st.sidebar.success(f"Cleared DB: text={res['deleted_text']} | media={res['deleted_media']}")
            st.session_state.messages = [
                {"role": "assistant", "content": "Database cleared. Upload docs again."}
            ]
            st.session_state.last_answer = None
            st.session_state.last_evaluation = None 
            st.rerun()


def render_header(system: RAGSystem):
    st.markdown('<div class="title">Local RAG with Chatty (Ollama)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Chat with your documents. Answer details on the right.</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Text chunks", system.text_db.count())
    with col2:
        st.metric("Media items", system.media_db.count())
    with col3:
        st.metric("Top K", system.config["top_k_retrieval"])


def upload_panel(system: RAGSystem):
    with st.expander("Upload and ingest files", expanded=True):
        uploaded = st.file_uploader(
            "Upload PDF, TXT/MD, images, or videos",
            type=["pdf", "txt", "md", "png", "jpg", "jpeg", "mp4", "mov", "mkv"],
            accept_multiple_files=True,
        )

        if uploaded and st.button("Save and ingest uploads"):
            total_added = 0
            for f in uploaded:
                try:
                    path = system.save_uploaded_file_to_data(f)
                    result = system.ingest_file(path)
                    total_added += result["chunks_added"]
                    st.success(
                        f"{f.name} ingested as {result['kind']} (items added: {result['chunks_added']})"
                    )
                except Exception as e:
                    st.error(f"Failed on {f.name}: {e}")

            st.info(f"Total items/chunks added: {total_added}")

        st.divider()

        if st.button("Ingest everything already inside ./data"):
            st.write(system.ingest_data_folder())


def link_panel(system: RAGSystem):
    with st.expander("Ingest links", expanded=False):
        links_raw = st.text_area("Paste links (one per line)", height=120)
        if st.button("Ingest links now"):
            links = [x.strip() for x in links_raw.split("\n") if x.strip()]
            total = 0
            for link in links:
                total += system.ingest_text_source(link)
            st.success(f"Added text chunks from links: {total}")


def sources_panel(system: RAGSystem):
    with st.expander("Manage sources", expanded=False):
        srcs = system.list_all_sources()
        all_sources = sorted(list(set(srcs["text_sources"] + srcs["media_sources"])))

        if not all_sources:
            st.write("No sources stored yet.")
            return

        selected = st.selectbox("Select a source to delete", all_sources)
        if st.button("Delete selected source"):
            res = system.delete_source_everywhere(selected)
            st.success(
                f"Deleted text chunks: {res['deleted_text']} | Deleted media items: {res['deleted_media']}"
            )
            st.session_state.last_answer = None
            st.session_state.last_evaluation = None 
            st.rerun()


def chat_feed():
    st.subheader("Chat")

    with st.container(border=True):
        for msg in st.session_state.messages:
            if msg["role"] == "assistant":
                with st.chat_message("assistant", avatar="🔷"):
                    st.markdown("**Chatty**")
                    if msg["content"] == "__THINKING__":
                        st.write("Thinking...")
                    else:
                        st.write(msg["content"])
            else:
                with st.chat_message("user", avatar="🧑‍💻"):
                    st.markdown("**User**")
                    st.write(msg["content"])


def chat_input_bar():
    with st.form("chat_form", clear_on_submit=True):
        colGear, colText, colSend = st.columns([0.10, 0.70, 0.20], gap="small")

        with colGear:
            with st.popover("⚙", use_container_width=True):
                st.markdown("**Answer settings**")
                st.session_state.ui_show_sources = st.checkbox(
                    "Show sources", value=st.session_state.ui_show_sources
                )
                st.session_state.ui_show_chunks = st.checkbox(
                    "Show retrieved chunks", value=st.session_state.ui_show_chunks
                )
                st.session_state.ui_show_media = st.checkbox(
                    "Show related media", value=st.session_state.ui_show_media
                )

        with colText:
            user_input = st.text_input(
                "Message",
                placeholder="Ask something about your documents...",
                label_visibility="collapsed",
            )

        with colSend:
            send = st.form_submit_button("Send", use_container_width=True)

    if send and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.messages.append({"role": "assistant", "content": "__THINKING__"})
        st.rerun()



def process_thinking(system: RAGSystem, RAG):
    if not st.session_state.messages:
        return
    last = st.session_state.messages[-1]
    if last["role"] != "assistant":
        return
    if last["content"] != "__THINKING__":
        return

    # find latest user message
    question = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            question = msg["content"]
            break

    if not question:
        st.session_state.messages[-1]["content"] = "I didn't receive a question."
        st.rerun()

    # init success/failure history
    if "success_cases" not in st.session_state:
        st.session_state.success_cases = []

    status_placeholder = st.empty()
    with status_placeholder:
        with st.chat_message("assistant", avatar="🔷"):
            st.markdown("**Chatty**")
            with st.status("Thinking...", expanded=False) as status:

                t0 = time.perf_counter()

                retrieved_chunks = []
                retrieval_time = 0.0

                status.update(label="Searching documents...", state="running")

                if RAG:
                    t_retrieval_start = time.perf_counter()
                    out = system.answer(question)
                    t_retrieval_end = time.perf_counter()

                    retrieval_time = t_retrieval_end - t_retrieval_start
                    retrieved_chunks = out.get("retrieved_chunks", [])
                else:
                    out = system.noRAGAnswer(question)

                answer = out["answer"]

                status.update(label="Evaluating answer quality...", state="running")
                evaluator = st.session_state.evaluator

                chunk_texts = [c.get("text", "") for c in retrieved_chunks]

                evaluation = evaluator.evaluate_response(
                    query=question,
                    answer=answer,
                    retrieved_chunks=chunk_texts
                )

                t1 = time.perf_counter()
                total_time = t1 - t0
                generation_time = max(0.0, total_time - retrieval_time)

                # attach time metrics
                evaluation["latency"] = {
                    "retrieval_sec": round(retrieval_time, 4),
                    "generation_sec": round(generation_time, 4),
                    "total_sec": round(total_time, 4),
                }

                # success / failure classification
                success = (evaluation["composite_score"] >= 0.7) and (not evaluation["hallucination"])
                evaluation["success"] = success

                # store case history (last 20)
                st.session_state.success_cases.append({
                    "question": question,
                    "score": round(evaluation["composite_score"], 3),
                    "hallucination": evaluation["hallucination"],
                    "success": success,
                    "rag_enabled": bool(RAG),
                    "latency": evaluation["latency"],
                })
                st.session_state.success_cases = st.session_state.success_cases[-20:]

                st.session_state.last_evaluation = evaluation

                status.update(label="Done!", state="complete")

    st.session_state.last_answer = out
    st.session_state.messages[-1]["content"] = answer
    status_placeholder.empty()
    st.rerun()



def answer_details_panel():
    st.subheader("Answer details")
    out = st.session_state.last_answer
    evaluation = st.session_state.last_evaluation
    
    if not out:
        st.write("Ask something to see sources and retrieval details.")
        return
    
    
    if evaluation:
        with st.expander("Answer Quality Metrics", expanded=True):

            st.metric(
                "Faithfulness",
                f"{evaluation['faithfulness']:.2%}",
                help="How well grounded the answer is in retrieved chunks"
            )

            st.metric(
                "Relevance",
                f"{evaluation['relevance']:.2%}",
                help="How relevant the answer is to the question"
            )

            st.metric(
                "Composite Score",
                f"{evaluation['composite_score']:.2%}",
                help="Overall answer quality score"
            )

            hallucination_status = "Yes" if evaluation["hallucination"] else "No"
            st.metric(
                "Hallucination",
                hallucination_status,
                help="Whether unsupported claims were detected"
            )

            # ---- Latency ----
            st.divider()
            latency = evaluation.get("latency", {})

            st.metric("Retrieval time (sec)", latency.get("retrieval_sec", 0.0))
            st.metric("Generation time (sec)", latency.get("generation_sec", 0.0))
            st.metric("Total time (sec)", latency.get("total_sec", 0.0))

            # ---- Success / Failure ----
            st.divider()
            success = evaluation.get("success", False)
            st.metric("Outcome", "Success" if success else "Failure")

            # Quality indicator (keep yours)
            if evaluation["composite_score"] >= 0.7:
                st.success("High quality answer")
            elif evaluation["composite_score"] >= 0.3:
                st.warning("Medium quality answer")
            else:
                st.error("Low quality answer")
        if st.session_state.ui_show_sources:
            with st.expander("Sources", expanded=True):
                sources = out.get("sources", [])
                if not sources:
                    st.write("No sources")
                else:
                    for s in sources:
                        st.write(s)
    
    if st.session_state.ui_show_chunks:
        with st.expander("Retrieved chunks", expanded=False):
            chunks = out.get("retrieved_chunks", [])
            if not chunks:
                st.write("No retrieved chunks")
            else:
                for i, ch in enumerate(chunks, start=1):
                    src = ch.get("metadata", {}).get("source", "unknown")
                    dist = ch.get("distance", None)
                    st.markdown(f"**Chunk {i}** | Source: `{src}` | Distance: `{dist:.4f}`")
                    st.text_area(f"chunk_{i}", ch.get("text", ""), height=100, label_visibility="collapsed")
                    st.divider()
    
    if st.session_state.ui_show_media:
        with st.expander("Related media", expanded=False):
            media = out.get("retrieved_media", [])
            if not media:
                st.write("No media results")
            else:
                for i, m in enumerate(media, start=1):
                    meta = m.get("metadata", {})
                    kind = meta.get("type", "media")
                    src = meta.get("source", "unknown")
                    dist = m.get("distance", None)
                    st.markdown(f"{i}. Type: {kind} | Source: {src} | Distance: {dist}")
                    st.write(m.get("document"))

def ragOnOff():
    ragButton = st.toggle("RAG")
    return ragButton

def main():
    st.set_page_config(page_title="Local RAG", layout="wide")
    apply_blue_theme()
    init_state()

    system = st.session_state.system

    sidebar_settings(system)
    rag = ragOnOff()
    render_header(system)

    left, middle, right = st.columns([0.32, 0.46, 0.22], gap="large")

    with left:
        upload_panel(system)
        link_panel(system)
        sources_panel(system)

    with middle:
        chat_feed()

        with st.container(border=True):
            chat_input_bar()

        process_thinking(system,rag)

    with right:
        with st.container(border=True):
            answer_details_panel()


if __name__ == "__main__":
    main()