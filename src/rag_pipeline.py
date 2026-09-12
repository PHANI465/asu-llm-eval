# =============================================================================
# src/rag_pipeline.py
# ASU LLM Evaluation — RAG Pipeline (Pinecone backend)
#
# Responsibilities:
#   1. Load .txt documents from data/knowledge_base/
#   2. Chunk them with configurable size / overlap
#   3. Embed with OpenAI text-embedding-3-small (1536 dims)
#   4. Store / retrieve vectors via the Pinecone cloud index "asullmeval"
#        - Each knowledge-base version gets its own namespace, named after a
#          hash of the chunks + embedding model
#        - Namespace already complete → connect directly (skip re-upload)
#        - Knowledge base edited      → new namespace is embedded + upserted,
#          so the evaluation never runs against stale vectors
#   5. Expose get_answer(question) -> dict with answer + metadata
# =============================================================================

import glob
import hashlib
import os
import time

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.schema import SystemMessage, HumanMessage
from pinecone import Pinecone

from project_config import KNOWLEDGE_BASE_DIR, load_config, require_env, token_cost_usd

# -----------------------------------------------------------------------------
# 0. Settings from config.yaml
# -----------------------------------------------------------------------------

_EVAL_CFG      = load_config()["evaluation"]
CHUNK_SIZE     = _EVAL_CFG["chunk_size"]           # 1000
CHUNK_OVERLAP  = _EVAL_CFG["chunk_overlap"]         # 150
TOP_K          = _EVAL_CFG["top_k_retrieval"]       # 8
EMBED_MODEL    = _EVAL_CFG["embedding_model"]       # text-embedding-3-small
LLM_MODEL      = _EVAL_CFG["model"]                 # gpt-4o
PINECONE_INDEX = _EVAL_CFG["pinecone_index"]        # asullmeval

# Seconds to wait for freshly upserted vectors to show up in index stats
_UPSERT_VISIBILITY_TIMEOUT = 60

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
    Raises if any file cannot be read — evaluating against a silently
    incomplete knowledge base would produce misleading scores.
    """
    print("\n[1/4] Loading documents from knowledge base...")

    txt_files = sorted(glob.glob(os.path.join(KNOWLEDGE_BASE_DIR, "*.txt")))
    if not txt_files:
        raise FileNotFoundError(
            f"No .txt files found in {KNOWLEDGE_BASE_DIR}. "
            "Make sure the knowledge base files are populated."
        )

    all_docs = []
    for path in txt_files:
        filename = os.path.basename(path)
        try:
            docs = TextLoader(path, encoding="utf-8").load()
        except Exception as e:
            raise RuntimeError(f"Failed to load knowledge base file {filename}: {e}") from e
        # Tag every chunk with the source filename for traceability
        for doc in docs:
            doc.metadata["source"] = filename
        all_docs.extend(docs)
        print(f"   [OK] Loaded: {filename}  ({len(docs[0].page_content)} chars)")

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
    print(f"   >> {len(chunks)} chunk(s) created.")
    return chunks


# -----------------------------------------------------------------------------
# 4. Pinecone vector store — one namespace per knowledge-base version
# -----------------------------------------------------------------------------

def knowledge_base_fingerprint(chunks) -> str:
    """
    Stable hash of everything that determines the stored vectors: the
    embedding model plus every chunk's source and text (chunk size/overlap
    are captured implicitly through the chunk texts).
    """
    h = hashlib.sha256(EMBED_MODEL.encode("utf-8"))
    for chunk in chunks:
        h.update(b"\x00" + chunk.metadata.get("source", "").encode("utf-8"))
        h.update(b"\x00" + chunk.page_content.encode("utf-8"))
    return h.hexdigest()


def _namespace_counts(index) -> dict:
    """Return {namespace: vector_count} from the index stats."""
    counts = {}
    for name, summary in (index.describe_index_stats().namespaces or {}).items():
        count = getattr(summary, "vector_count", None)
        if count is None and isinstance(summary, dict):
            count = summary.get("vector_count", 0)
        counts[name] = count or 0
    return counts


def build_or_load_vectorstore(chunks):
    """
    Connect to the Pinecone index and return a LangChain PineconeVectorStore
    bound to the namespace for the current knowledge-base version.

    - Namespace already holds every chunk → reuse it (no embedding cost).
    - Otherwise → embed and upsert all chunks with deterministic IDs
      (idempotent, so a partially uploaded namespace is simply completed).
    """
    embeddings = OpenAIEmbeddings(
        model=EMBED_MODEL,
        openai_api_key=require_env("OPENAI_API_KEY"),
    )

    namespace = f"kb-{knowledge_base_fingerprint(chunks)[:16]}"
    print(f"\n[3/4] Connecting to Pinecone index '{PINECONE_INDEX}' (namespace '{namespace}')")

    pc    = Pinecone(api_key=require_env("PINECONE_API_KEY"))
    index = pc.Index(PINECONE_INDEX)
    vectorstore = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)

    counts   = _namespace_counts(index)
    existing = counts.get(namespace, 0)

    if existing == len(chunks):
        print(f"   Knowledge base unchanged — {existing} vectors already indexed.")
    else:
        print(f"   New or changed knowledge base ({existing}/{len(chunks)} vectors present) "
              f"— uploading {len(chunks)} vectors...")
        vectorstore.add_documents(chunks, ids=[f"chunk-{i:04d}" for i in range(len(chunks))])

        # Serverless indexes are eventually consistent — wait until the
        # vectors are visible so the first questions don't retrieve nothing.
        deadline = time.monotonic() + _UPSERT_VISIBILITY_TIMEOUT
        while _namespace_counts(index).get(namespace, 0) < len(chunks):
            if time.monotonic() > deadline:
                print("   [WARNING] Upserted vectors are not fully visible yet — continuing anyway.")
                break
            time.sleep(2)
        print("   Vectors uploaded successfully.")

    stale = sorted(ns or "(default)" for ns in counts if ns != namespace)
    if stale:
        print(f"   Note: {len(stale)} namespace(s) from older knowledge-base versions are unused "
              f"and can be deleted in the Pinecone console: {', '.join(stale)}")

    return vectorstore


# -----------------------------------------------------------------------------
# 5. Lazy singletons — built once per process
# -----------------------------------------------------------------------------

_vectorstore = None
_llm         = None


def _get_vectorstore():
    """Return the shared vectorstore, initialising it on first call."""
    global _vectorstore
    if _vectorstore is None:
        docs   = load_documents()
        chunks = split_documents(docs)
        _vectorstore = build_or_load_vectorstore(chunks)
    return _vectorstore


def _get_llm():
    """Return the shared answer model client, initialising it on first call."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0,
            max_retries=3,
            openai_api_key=require_env("OPENAI_API_KEY"),
        )
    return _llm


# -----------------------------------------------------------------------------
# 6. Main public API — get_answer()
# -----------------------------------------------------------------------------

def get_answer(question: str) -> dict:
    """
    Given a question string, retrieve the top-K relevant chunks from Pinecone
    and generate an answer with the configured LLM.

    Never raises: failures (bad key, exhausted quota, network) are reported in
    the "error" field so callers can tell a failed call from a real answer.

    Returns
    -------
    dict with keys:
        question         – original question
        answer           – LLM-generated answer ("" when the call failed)
        retrieved_chunks – list of chunk texts used as context
        source_documents – list of source filenames
        latency_seconds  – wall-clock time for the full call
        token_usage      – total tokens consumed (input + output)
        input_tokens     – prompt tokens
        output_tokens    – completion tokens
        cost_usd         – call cost from config.yaml pricing (None if unpriced)
        error            – None on success, otherwise "ExceptionType: message"
    """
    start_time = time.perf_counter()
    result = {
        "question":         question,
        "answer":           "",
        "retrieved_chunks": [],
        "source_documents": [],
        "latency_seconds":  0.0,
        "token_usage":      0,
        "input_tokens":     0,
        "output_tokens":    0,
        "cost_usd":         None,
        "error":            None,
    }

    try:
        # --- Retrieve top-K chunks ---
        relevant_docs = _get_vectorstore().similarity_search(question, k=TOP_K)

        retrieved_chunks = [doc.page_content for doc in relevant_docs]
        source_documents = [doc.metadata.get("source", "unknown") for doc in relevant_docs]

        # --- Build context block for the prompt ---
        context_block = "\n\n---\n\n".join(
            f"[Source: {src}]\n{chunk}"
            for src, chunk in zip(source_documents, retrieved_chunks)
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context_block}\n\nQuestion: {question}"),
        ]

        response = _get_llm().invoke(messages)
        usage    = getattr(response, "usage_metadata", None) or {}

        input_tokens  = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        result.update({
            "answer":           response.content,
            "retrieved_chunks": retrieved_chunks,
            "source_documents": source_documents,
            "token_usage":      usage.get("total_tokens", input_tokens + output_tokens),
            "input_tokens":     input_tokens,
            "output_tokens":    output_tokens,
            "cost_usd":         token_cost_usd(LLM_MODEL, input_tokens, output_tokens),
        })

    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    result["latency_seconds"] = round(time.perf_counter() - start_time, 3)
    return result


# -----------------------------------------------------------------------------
# 7. Quick-test entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  ASU RAG Pipeline — Quick Test (Pinecone backend)")
    print("=" * 60)

    TEST_QUESTIONS = [
        "What is the minimum GPA for undergraduate admission?",
        "What is the tuition for international students?",
        "What meal plans are available at ASU?",
    ]

    print("\n[4/4] Running test queries...\n")

    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"--- Question {i} -------------------------------------------")
        result = get_answer(q)
        print(f"Q: {result['question']}")
        if result["error"]:
            print(f"   ERROR   : {result['error']}")
        else:
            print(f"A: {result['answer']}")
            print(f"   Sources : {result['source_documents']}")
            print(f"   Latency : {result['latency_seconds']}s")
            print(f"   Tokens  : {result['token_usage']}  (cost ${result['cost_usd'] or 0:.5f})")
        print()

    print("=" * 60)
    print("  Pipeline test complete.")
    print("=" * 60)
