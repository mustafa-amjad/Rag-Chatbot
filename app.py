import hashlib
import os
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from rag import pipeline
from rag.db import Store

load_dotenv()
DATA_DIR = Path("rag_data")

st.set_page_config(page_title="RAG Assistant", page_icon=":material/menu_book:", layout="wide")


def server_key() -> str:
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:  # no secrets file locally
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


@st.cache_resource
def get_store() -> Store:
    return Store(DATA_DIR / "sessions.db")


@st.cache_resource(show_spinner="Loading the embedding model...")
def get_vectors():
    return pipeline.vector_store(str(DATA_DIR / "chroma"), pipeline.embeddings())


@st.cache_resource
def get_llm(api_key: str):
    return pipeline.llm(api_key)


# Each visitor gets a private workspace kept in the URL, so the link can be bookmarked.
if "ws" not in st.query_params:
    st.query_params["ws"] = uuid.uuid4().hex[:12]
workspace = st.query_params["ws"]
db = get_store()

with st.sidebar:
    st.subheader("Sessions")
    with st.form("new_session", clear_on_submit=True, border=False):
        name = st.text_input("New session", placeholder="e.g. Q3 finance reports")
        if st.form_submit_button("Create session", icon=":material/add:", width="stretch") and name.strip():
            st.session_state.sid = db.create_session(workspace, name.strip())

    sessions = db.sessions(workspace)
    ids = [s[0] for s in sessions]
    if st.session_state.get("sid") not in ids:
        st.session_state.sid = ids[0] if ids else None
    if sessions:
        labels = dict(sessions)
        st.session_state.sid = st.radio(
            "Your sessions", ids, index=ids.index(st.session_state.sid),
            format_func=labels.get, label_visibility="collapsed",
        )

    with st.expander("Retrieval settings", icon=":material/tune:"):
        top_k = st.slider("Chunks per answer", 2, 12, 5)
        chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1200, 100)
        overlap = st.slider("Chunk overlap", 50, 400, 150, 50)

    key = server_key()
    if not key:
        key = st.text_input("Groq API key", type="password", help="Free key at console.groq.com")

    st.caption(
        "Your sessions are private to this link. Bookmark it to come back later.  \n"
        "Built by Mustafa Amjad · [Source](https://github.com/mustafa-amjad/Rag-Chatbot)"
    )

st.title("RAG Assistant")
st.caption("Ask questions about your PDFs. Answers come only from your documents, with page-level sources.")

if not key:
    st.info("Add a Groq API key in the sidebar to start.", icon=":material/key:")
    st.stop()
sid = st.session_state.sid
if sid is None:
    st.info("Create a session in the sidebar to begin.", icon=":material/arrow_back:")
    st.stop()

session_name = dict(sessions)[sid]
store = get_vectors()

head = st.container(horizontal=True, vertical_alignment="center")
head.subheader(session_name)
if head.button("Delete session", icon=":material/delete:", type="tertiary"):
    db.delete_session(workspace, sid)
    st.session_state.sid = None
    st.rerun()

with st.expander("Documents", icon=":material/folder_open:", expanded=not db.files(sid)):
    uploads = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key=f"up_{sid}")
    for f in uploads or []:
        data = f.getvalue()
        file_hash = hashlib.md5(data).hexdigest()
        if db.has_file(sid, file_hash):
            continue
        with st.spinner(f"Indexing {f.name}..."):
            pages, chunks = pipeline.split_pdf(data, f.name, sid, chunk_size, overlap)
            if chunks:
                store.add_documents(chunks)
            db.add_file(sid, f.name, file_hash, pages, len(chunks))
        st.toast(f"Indexed {f.name}: {pages} pages, {len(chunks)} chunks", icon=":material/check:")
    files = db.files(sid)
    if files:
        st.dataframe(
            [{"File": n, "Pages": p, "Chunks": c} for n, p, c in files],
            hide_index=True,
        )

if not db.files(sid):
    st.caption("Upload at least one PDF to start asking questions.")
    st.stop()

history = db.messages(sid)
for role, content in history:
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("Ask about your documents", submit_mode="disable")
if question:
    with st.chat_message("user"):
        st.markdown(question)
    model = get_llm(key)
    with st.chat_message("assistant"):
        try:
            query, docs = pipeline.retrieve(store, model, sid, question, history, top_k)
            answer = st.write_stream(pipeline.stream_answer(model, question, history, docs))
        except Exception as e:
            st.error(f"Something went wrong: {e}", icon=":material/error:")
            st.stop()
        src = pipeline.sources(docs)
        if src:
            st.caption("Sources: " + " · ".join(src))
        with st.expander("Retrieved passages", icon=":material/search:"):
            st.caption(f"Search query: {query}")
            for d in docs:
                st.markdown(f"**{d.metadata.get('source_file')}**, page {d.metadata.get('page', 0) + 1}")
                st.text(d.page_content[:600])
    db.add_message(sid, "user", question)
    db.add_message(sid, "assistant", answer)

if history and st.button("Clear chat history", icon=":material/delete_sweep:", type="tertiary"):
    db.clear_messages(sid)
    st.rerun()
