"""Indexing and retrieval-augmented answering."""
from __future__ import annotations

import os
import tempfile
from typing import Iterator

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"
NOT_FOUND = "I couldn't find that in the uploaded documents."
MAX_CONTEXT_CHARS = 7000

REWRITE = ChatPromptTemplate.from_messages([
    ("system", "Rewrite the user's latest question as a standalone search query, using the "
               "conversation for context. Output only the query."),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])

ANSWER = ChatPromptTemplate.from_messages([
    ("system",
     "You answer questions using only the context below.\n"
     f"- If the answer is not in the context, reply exactly: \"{NOT_FOUND}\"\n"
     "- Cite the source file and page when you can.\n"
     "- Be concise. Use bullet points for lists.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


def vector_store(persist_dir: str, emb: HuggingFaceEmbeddings) -> Chroma:
    return Chroma(collection_name="documents", persist_directory=persist_dir, embedding_function=emb)


def llm(api_key: str) -> ChatGroq:
    return ChatGroq(api_key=api_key, model=LLM_MODEL, temperature=0.1)


def split_pdf(data: bytes, filename: str, session_id: str, chunk_size: int, overlap: int) -> tuple[int, list[Document]]:
    """Return (page count, chunks) for one PDF, tagged with its session."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        pages = PyPDFLoader(path).load()
    finally:
        os.unlink(path)
    for p in pages:
        p.metadata.update(source_file=filename, session_id=session_id)
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return len(pages), splitter.split_documents(pages)


def _history(messages: list[tuple[str, str]]):
    return [HumanMessage(c) if r == "user" else AIMessage(c) for r, c in messages]


def retrieve(store: Chroma, model: ChatGroq, session_id: str, question: str,
             messages: list[tuple[str, str]], k: int) -> tuple[str, list[Document]]:
    """Rewrite the question with history, then fetch diverse chunks from this session only."""
    history = _history(messages)
    query = model.invoke(REWRITE.format_messages(history=history, question=question)).content.strip()
    docs = store.max_marginal_relevance_search(query, k=k, fetch_k=k * 4, filter={"session_id": session_id})
    return query, docs


def context_text(docs: list[Document]) -> str:
    parts, total = [], 0
    for d in docs:
        if total + len(d.page_content) > MAX_CONTEXT_CHARS:
            break
        page = d.metadata.get("page")
        label = f"[{d.metadata.get('source_file', '?')}, page {page + 1 if isinstance(page, int) else '?'}]"
        parts.append(f"{label}\n{d.page_content}")
        total += len(d.page_content)
    return "\n\n---\n\n".join(parts)


def stream_answer(model: ChatGroq, question: str, messages: list[tuple[str, str]],
                  docs: list[Document]) -> Iterator[str]:
    prompt = ANSWER.format_messages(history=_history(messages), question=question, context=context_text(docs))
    for chunk in model.stream(prompt):
        yield chunk.content


def sources(docs: list[Document]) -> list[str]:
    seen = []
    for d in docs:
        page = d.metadata.get("page")
        s = f"{d.metadata.get('source_file', '?')} · page {page + 1 if isinstance(page, int) else '?'}"
        if s not in seen:
            seen.append(s)
    return seen
