# =============================================================================
# src/rag_pipeline.py
# ASU LLM Evaluation — RAG Pipeline
#
# Responsibilities:
#   1. Load .txt documents from data/knowledge_base/
#   2. Chunk them with configurable size / overlap
#   3. Embed with OpenAI text-embedding-3-small
#   4. Persist vectors in ChromaDB (skip rebuild if already exists)
#   5. Expose get_answer(question) -> dict with answer + metadata
# =============================================================================

import os
import time
import glob
import yaml

from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.schema import SystemMessage, HumanMessage

# -----------------------------------------------------------------------------
# 0. Bootstrap — load env vars and project config
# -----------------------------------------------------------------------------

# Load OPENAI_API_KEY from .env (project root)
_ENV_PATH = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found. "
        "Make sure it is set in the .env file at the project root."
    )

# Load config.yaml
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _CONFIG = yaml.safe_load(_f)

_EVAL_CFG = _CONFIG["evaluation"]
CHUNK_SIZE        = _EVAL_CFG["chunk_size"]          # 1000
CHUNK_OVERLAP     = _EVAL_CFG["chunk_overlap"]        # 150
TOP_K             = _EVAL_CFG["top_k_retrieval"]      # 5
EMBED_MODEL       = _EVAL_CFG["embedding_model"]      # text-embedding-3-small
LLM_MODEL         = _EVAL_CFG["model"]                # gpt-4o

# Paths
_PROJECT_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
KNOWLEDGE_BASE   = os.path.join(_PROJECT_ROOT, "data", "knowledge_base")
CHROMA_DB_DIR    = os.path.join(_PROJECT_ROOT, "chroma_db")

# -----------------------------------------------------------------------------
# 1. System prompt for the LLM
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful ASU university assistant. "
    "Answer questions using ONLY the provided context documents. "
    "If the answer is not in the context, say "
    "'I don't have information about that in my knowledge base.' "
    "Do not make up information. Be concise and accurate. "
    "IMPORTANT: When a question asks about a score threshold (SAT, ACT, GPA) "
    "for a test that is described as optional in the context, still provide "
    "the specific threshold number and clarify that submitting is optional. "
    "For example: 'The SAT threshold for non-residents is 1180 if submitting scores, "
    "though SAT submission is optional.' Never refuse to state a threshold simply "
    "because the test is optional."
)

# -----------------------------------------------------------------------------
# 2. Document loading
# -----------------------------------------------------------------------------

def load_documents():
    """
    Load every .txt file from data/knowledge_base/.
    Returns a list of LangChain Document objects with metadata.
    """
    print("\n[1/4] Loading documents from knowledge base...")

    txt_files = sorted(glob.glob(os.path.join(KNOWLEDGE_BASE, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {KNOWLEDGE_BASE}. "
            "Make sure the knowledge base files are populated."
        )

    all_docs = []
    for path in txt_files:
        filename = os.path.basename(path)
        try:
            loader = TextLoader(path, encoding="utf-8")
            docs = loader.load()
            # Tag every page with the source filename for easy traceability
            for doc in docs:
                doc.metadata["source"] = filename
            all_docs.extend(docs)
            print(f"   [OK] Loaded: {filename}  ({len(docs[0].page_content)} chars)")
        except Exception as e:
            print(f"   [FAIL] Failed to load {filename}: {e}")

    print(f"   >> {len(all_docs)} document(s) loaded from {len(txt_files)} file(s).")
    return all_docs

# -----------------------------------------------------------------------------
# 3. Chunking
# -----------------------------------------------------------------------------

def split_documents(documents):
    """
    Split documents into overlapping chunks using RecursiveCharacterTextSplitter.
    chunk_size / chunk_overlap are read from config.yaml.
    """
    print(f"\n[2/4] Splitting documents (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"   >>{len(chunks)} chunk(s) created.")
    return chunks

# -----------------------------------------------------------------------------
# 4. Embeddings + ChromaDB (create or load)
# -----------------------------------------------------------------------------

def build_or_load_vectorstore(chunks=None):
    """
    If ./chroma_db already contains data, load it directly.
    Otherwise embed the provided chunks and persist a new ChromaDB store.
    Returns a LangChain Chroma vectorstore instance.
    """
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )

    # Detect whether the DB has already been built
    db_populated = (
        os.path.isdir(CHROMA_DB_DIR)
        and any(True for _ in os.scandir(CHROMA_DB_DIR))
    )

    if db_populated:
        print(f"\n[3/4] ChromaDB found at '{CHROMA_DB_DIR}' — loading existing store.")
        vectorstore = Chroma(
            persist_directory=CHROMA_DB_DIR,
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        print(f"   >>{count} vector(s) already stored. Skipping re-embedding.")
    else:
        if chunks is None:
            raise ValueError("chunks must be provided when ChromaDB does not exist yet.")
        print(f"\n[3/4] Creating embeddings & persisting to '{CHROMA_DB_DIR}'...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DB_DIR,
        )
        count = vectorstore._collection.count()
        print(f"   >>{count} vector(s) embedded and stored.")

    return vectorstore

# -----------------------------------------------------------------------------
# 5. Module-level vectorstore (lazy singleton — built once per process)
# -----------------------------------------------------------------------------

_vectorstore = None

def _get_vectorstore():
    """Return the shared vectorstore, initialising it on first call."""
    global _vectorstore
    if _vectorstore is None:
        docs   = load_documents()
        chunks = split_documents(docs)
        _vectorstore = build_or_load_vectorstore(chunks)
    return _vectorstore

# -----------------------------------------------------------------------------
# 6. Main public API — get_answer()
# -----------------------------------------------------------------------------

def get_answer(question: str) -> dict:
    """
    Given a question string, retrieve the top-K relevant chunks from ChromaDB
    and generate an answer using GPT-4o.

    Returns
    -------
    dict with keys:
        question         – original question
        answer           – LLM-generated answer
        retrieved_chunks – list of chunk texts used as context
        source_documents – list of source filenames
        latency_seconds  – wall-clock time for the full call
        token_usage      – total tokens consumed (prompt + completion)
    """
    start_time = time.perf_counter()

    try:
        vs = _get_vectorstore()

        # --- Retrieve top-K chunks ---
        retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
        relevant_docs = retriever.invoke(question)

        retrieved_chunks   = [doc.page_content for doc in relevant_docs]
        source_documents   = [doc.metadata.get("source", "unknown") for doc in relevant_docs]

        # --- Build context block for the prompt ---
        context_block = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}"
            for src, chunk in zip(source_documents, retrieved_chunks)
        )

        # --- Call GPT-4o ---
        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            openai_api_key=OPENAI_API_KEY,
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Context:\n{context_block}\n\n"
                    f"Question: {question}"
                )
            ),
        ]

        response = llm.invoke(messages)

        answer      = response.content
        token_usage = (
            response.usage_metadata.get("total_tokens", 0)
            if hasattr(response, "usage_metadata") and response.usage_metadata
            else 0
        )

    except Exception as e:
        answer           = f"Error generating answer: {e}"
        retrieved_chunks = []
        source_documents = []
        token_usage      = 0

    latency = round(time.perf_counter() - start_time, 3)

    return {
        "question":         question,
        "answer":           answer,
        "retrieved_chunks": retrieved_chunks,
        "source_documents": source_documents,
        "latency_seconds":  latency,
        "token_usage":      token_usage,
    }

# -----------------------------------------------------------------------------
# 7. Quick-test entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  ASU RAG Pipeline - Quick Test")
    print("=" * 60)

    TEST_QUESTIONS = [
        "What is the minimum GPA required for undergraduate admission to ASU?",
        "What is the tuition for an international undergraduate student at ASU?",
        "What meal plans are available at ASU and how much do they cost?",
    ]

    print("\n[4/4] Running test queries...\n")

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"--- Question {i} -------------------------------------------")
        result = get_answer(q)
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")
        print(f"   Sources : {result['source_documents']}")
        print(f"   Latency : {result['latency_seconds']}s")
        print(f"   Tokens  : {result['token_usage']}")
        print()

    print("=" * 60)
    print("  Pipeline test complete.")
    print("=" * 60)
