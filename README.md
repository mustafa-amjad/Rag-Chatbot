# RAG Assistant Pro

Chat with your PDFs. Upload documents, ask questions in plain English, and get answers grounded in those documents, with the file name and page number for every source.

Built with Streamlit, LangChain, ChromaDB, Hugging Face embeddings and Groq (Llama 3.3 70B).

## What it does

- **Upload PDFs** and index them into a persistent ChromaDB vector store. Files are hashed, so the same PDF is never indexed twice.
- **Rewrites the question** using the chat history before searching, so follow-ups like "what about the second one?" still find the right text.
- **Retrieves with MMR** (max marginal relevance) to get relevant chunks that aren't near-duplicates of each other.
- **Answers only from the documents.** If the answer isn't there, it says "Out of scope" instead of guessing.
- **Shows sources** (file and page) and the retrieved chunks, so every answer can be checked.
- **Persistent sessions** in SQLite: create, switch and delete chat sessions, and keep history across restarts.
- **Tunable retrieval:** chunk size, overlap and top-k are adjustable from the sidebar.

## How it works

```
PDF ──► PyPDFLoader ──► RecursiveCharacterTextSplitter ──► MiniLM embeddings ──► ChromaDB
                                                                                    │
Question ──► query rewrite (LLM + history) ──► MMR retrieval ◄──────────────────────┘
                                                    │
                                   context + history ──► Llama 3.3 70B (Groq) ──► answer + sources
```

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| LLM | Groq, `llama-3.3-70b-versatile` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent) |
| Sessions and history | SQLite |

## Run it locally

```bash
git clone https://github.com/mustafa-amjad/Rag-Chatbot.git
cd Rag-Chatbot
python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # then add your Groq API key
streamlit run Assignment12_MustafaAmjad_RAG_CHATBOT.py
```

You can also paste the Groq key into the sidebar instead of using `.env`. A sample PDF to try is in [`samples/`](samples/).

## Project structure

```
Rag-Chatbot/
├── Assignment12_MustafaAmjad_RAG_CHATBOT.py   # Streamlit app
├── samples/                                   # sample PDF for testing
├── requirements.txt
├── packages.txt / runtime.txt                 # Streamlit Cloud config
├── .env.example
└── LICENSE
```

## Author

**Mustafa Amjad**, AI Engineer · [LinkedIn](https://www.linkedin.com/in/mustafaamjad9)

MIT License
