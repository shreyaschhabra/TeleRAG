"""
╔══════════════════════════════════════════════════════════════════╗
║          3GPP RAG Assistant — Release 16 & 18 Expert           ║
║──────────────────────────────────────────────────────────────────║
║  A production-ready Retrieval-Augmented Generation pipeline     ║
║  for parsing, indexing, and querying 3GPP technical specs.      ║
║                                                                  ║
║  Stack: LangChain · ChromaDB · HuggingFace · Google Gemini     ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Standard Library ─────────────────────────────────────────────
import os
import sys
import json
import glob
import hashlib
import time
from pathlib import Path

# ── Fix Windows Console Encoding ─────────────────────────────────
# Windows terminals default to cp1252 which cannot render Unicode
# characters (emoji, box-drawing). Force UTF-8 output encoding.
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ── Third-Party ──────────────────────────────────────────────────
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document

# ── Load Environment Variables ───────────────────────────────────
load_dotenv()

# ╔════════════════════════════════════════════════════════════════╗
# ║                    CONFIGURATION CONSTANTS                    ║
# ╚════════════════════════════════════════════════════════════════╝

# ── Paths ────────────────────────────────────────────────────────
# Folder containing the 3GPP JSON files  (adjust if needed)
JSON_DATA_DIR = os.path.dirname(os.path.abspath(__file__))

# Persistent ChromaDB directory
CHROMA_PERSIST_DIR = os.path.join(JSON_DATA_DIR, "chroma_db")

# ── Embedding Model ─────────────────────────────────────────────
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ── Chunking Parameters ─────────────────────────────────────────
CHUNK_SIZE = 800        # Characters per chunk
CHUNK_OVERLAP = 150     # Overlap between consecutive chunks

# ── Retriever Parameters ─────────────────────────────────────────
SEARCH_TYPE = "mmr"             # Maximal Marginal Relevance
MMR_FETCH_K = 20                # Candidates to consider for diversity
MMR_K = 6                       # Final documents to return
MMR_LAMBDA_MULT = 0.7           # 0 = max diversity, 1 = max relevance

# ── LLM Configuration ───────────────────────────────────────────
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_TEMPERATURE = 0.2        # Low temperature for factual answers
GEMINI_MAX_TOKENS = 2048

# ── JSON Field Mapping ──────────────────────────────────────────
# Adjust these keys if your JSON schema changes.
CONTENT_KEY = "text"           # The field containing specification text
METADATA_KEYS = {
    "id": "doc_id",            # Unique document identifier
    "spec_num": "spec_number", # 3GPP specification number
    "series": "series",        # Series classification
    "filename": "filename",    # Original filename
    "num_paragraphs": "num_paragraphs",
    "num_chars": "num_chars",
}


# ╔════════════════════════════════════════════════════════════════╗
# ║              1. JSON DATA INGESTION & PARSING                 ║
# ╚════════════════════════════════════════════════════════════════╝

def detect_release(filename: str, text: str) -> str:
    """
    Heuristically detect the 3GPP release number from the filename
    or the first 500 characters of the document text.
    
    Convention:
      - Files named '3gpp_16.json' → Release 16
      - Files named '3gpp_18.json' → Release 18
      - Falls back to scanning text for "(Release XX)"
    """
    fname_lower = filename.lower()
    if "16" in fname_lower:
        return "16"
    elif "18" in fname_lower:
        return "18"

    # Fallback: scan the document text itself
    snippet = text[:500].lower()
    if "release 18" in snippet:
        return "18"
    elif "release 16" in snippet:
        return "16"
    elif "release 17" in snippet:
        return "17"
    return "Unknown"


def load_json_documents(data_dir: str) -> list[Document]:
    """
    Iterate through all JSON files in `data_dir`, extract content
    and metadata, and return a list of LangChain Document objects.

    Each JSON file is expected to be a JSON array of objects.
    The content field and metadata fields are controlled by the
    CONTENT_KEY and METADATA_KEYS constants at the top of this file.

    Returns:
        List[Document]: Parsed LangChain documents with metadata.
    """
    json_files = sorted(glob.glob(os.path.join(data_dir, "*.json")))

    if not json_files:
        print(f"[ERROR] No JSON files found in: {data_dir}")
        sys.exit(1)

    # Filter to only 3GPP specification files (skip QnA/evaluation sets)
    spec_files = [f for f in json_files if "3gpp" in os.path.basename(f).lower()]

    if not spec_files:
        print("[WARNING] No files matching '3gpp*.json' found. Using all JSON files.")
        spec_files = json_files

    print(f"\n{'─'*60}")
    print(f"  📂 Found {len(spec_files)} specification file(s) to ingest")
    print(f"{'─'*60}")

    all_documents: list[Document] = []

    for json_path in spec_files:
        fname = os.path.basename(json_path)
        print(f"\n  ▸ Loading: {fname}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both list-of-objects and dict-of-objects formats
        if isinstance(data, dict):
            entries = list(data.values())
        elif isinstance(data, list):
            entries = data
        else:
            print(f"    [SKIP] Unexpected JSON root type: {type(data)}")
            continue

        print(f"    ✓ {len(entries):,} entries found")

        for entry in tqdm(entries, desc=f"    Parsing {fname}", unit="doc"):
            # ── Extract content ──────────────────────────────
            content = entry.get(CONTENT_KEY, "")
            if not content or not content.strip():
                continue  # Skip empty documents

            # ── Extract metadata ─────────────────────────────
            metadata = {}
            for json_key, meta_name in METADATA_KEYS.items():
                metadata[meta_name] = entry.get(json_key, "N/A")

            # ── Detect release number ────────────────────────
            metadata["release"] = detect_release(fname, content)

            # ── Source reference ─────────────────────────────
            metadata["source_file"] = fname

            all_documents.append(
                Document(page_content=content, metadata=metadata)
            )

    print(f"\n  ✅ Total documents loaded: {len(all_documents):,}")
    return all_documents


# ╔════════════════════════════════════════════════════════════════╗
# ║                   2. SMART CHUNKING                           ║
# ╚════════════════════════════════════════════════════════════════╝

def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents into smaller chunks using RecursiveCharacterTextSplitter.
    
    Because the 3GPP data is already semi-structured (coming from parsed
    spec documents), we use a chunk size of 800 with 150 overlap to
    preserve context across chunk boundaries.

    Metadata is automatically propagated to every chunk by LangChain.

    Returns:
        List[Document]: Chunked documents with metadata preserved.
    """
    print(f"\n{'─'*60}")
    print(f"  ✂️  Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"{'─'*60}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],  # Prefer paragraph → sentence → word
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(documents)
    print(f"  ✅ Created {len(chunks):,} chunks from {len(documents):,} documents")

    return chunks


# ╔════════════════════════════════════════════════════════════════╗
# ║               3. VECTOR STORE MANAGEMENT                      ║
# ╚════════════════════════════════════════════════════════════════╝

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize the HuggingFace embedding model.
    Uses all-MiniLM-L6-v2 — a fast, lightweight model ideal for
    semantic search without GPU or API costs.
    """
    print(f"\n  🔤 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
    print(f"  ✅ Embedding model ready")
    return embeddings


def build_or_load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    data_dir: str,
) -> Chroma:
    """
    Smart vector store management:
      - If a persisted ChromaDB exists at CHROMA_PERSIST_DIR → load it.
      - Otherwise → parse JSONs, chunk, embed, and persist a new store.

    Returns:
        Chroma: A ready-to-query ChromaDB vector store.
    """
    chroma_exists = (
        os.path.isdir(CHROMA_PERSIST_DIR)
        and len(os.listdir(CHROMA_PERSIST_DIR)) > 0
    )

    if chroma_exists:
        # ── Load existing vector store ───────────────────────
        print(f"\n{'═'*60}")
        print(f"  📦 Existing ChromaDB found at: {CHROMA_PERSIST_DIR}")
        print(f"  📦 Loading persisted vector store…")
        print(f"{'═'*60}")

        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings,
            collection_name="3gpp_specs",
        )

        # Quick health check
        collection_count = vectorstore._collection.count()
        print(f"  ✅ Loaded {collection_count:,} vectors from disk")

        return vectorstore

    else:
        # ── Build new vector store ───────────────────────────
        print(f"\n{'═'*60}")
        print(f"  🏗️  No existing ChromaDB found. Building from scratch…")
        print(f"{'═'*60}")

        # Step 1: Ingest JSON documents
        documents = load_json_documents(data_dir)

        # Step 2: Chunk documents
        chunks = chunk_documents(documents)

        # Step 3: Create and persist vector store
        print(f"\n{'─'*60}")
        print(f"  💾 Embedding & storing {len(chunks):,} chunks into ChromaDB…")
        print(f"     (This may take several minutes on first run)")
        print(f"{'─'*60}")

        # Process in batches to avoid memory issues with large datasets
        BATCH_SIZE = 500
        vectorstore = None

        for i in tqdm(range(0, len(chunks), BATCH_SIZE),
                      desc="  Embedding batches", unit="batch"):
            batch = chunks[i : i + BATCH_SIZE]

            if vectorstore is None:
                # First batch: create the vector store
                vectorstore = Chroma.from_documents(
                    documents=batch,
                    embedding=embeddings,
                    persist_directory=CHROMA_PERSIST_DIR,
                    collection_name="3gpp_specs",
                )
            else:
                # Subsequent batches: add to existing store
                vectorstore.add_documents(documents=batch)

        print(f"\n  ✅ ChromaDB persisted to: {CHROMA_PERSIST_DIR}")
        print(f"  ✅ Total vectors stored: {vectorstore._collection.count():,}")

        return vectorstore


# ╔════════════════════════════════════════════════════════════════╗
# ║               4. ADVANCED RETRIEVER (MMR)                     ║
# ╚════════════════════════════════════════════════════════════════╝

def create_retriever(vectorstore: Chroma):
    """
    Create an MMR (Maximal Marginal Relevance) retriever.
    
    MMR balances relevance and diversity: it first retrieves `fetch_k`
    candidates by similarity, then iteratively selects `k` documents
    that are both relevant to the query and diverse from each other.
    
    This prevents the retriever from returning near-duplicate chunks
    from the same section of a specification.
    """
    retriever = vectorstore.as_retriever(
        search_type=SEARCH_TYPE,
        search_kwargs={
            "k": MMR_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA_MULT,
        },
    )
    print(f"\n  🔍 MMR Retriever configured (k={MMR_K}, fetch_k={MMR_FETCH_K}, λ={MMR_LAMBDA_MULT})")
    return retriever


# ╔════════════════════════════════════════════════════════════════╗
# ║             5. QA CHAIN & SYSTEM PROMPT                       ║
# ╚════════════════════════════════════════════════════════════════╝

# ── System Prompt ────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a 3GPP Technical Standards Expert with deep knowledge of 
Release 16 and Release 18 specifications. Your role is to provide precise, 
technically accurate answers to questions about 3GPP standards.

STRICT RULES:
1. Answer the user's question using ONLY the provided context from 
   Release 16 and Release 18 documents.
2. If the context does not contain sufficient information to answer 
   the question, state explicitly: "This information is not covered 
   in the retrieved Release 16/18 documents."
3. When referencing specific standards, always cite the specification 
   number (e.g., TS 23.501) and section where applicable.
4. Use precise technical terminology consistent with 3GPP standards.
5. Structure your answers clearly with bullet points or numbered lists 
   when explaining multiple aspects.
6. If there are differences between Release 16 and Release 18 on the 
   same topic, highlight them explicitly.

CONTEXT FROM 3GPP SPECIFICATIONS:
{context}

USER QUESTION:
{question}

EXPERT ANSWER:"""


def build_qa_chain(retriever, llm):
    """
    Build a RetrievalQA chain that:
      1. Retrieves relevant chunks via MMR.
      2. Passes them as context to the Gemini LLM.
      3. Returns both the answer and source documents.
    """
    prompt = PromptTemplate(
        template=SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",           # Stuff all context into one prompt
        retriever=retriever,
        return_source_documents=True,  # Required for explainability
        chain_type_kwargs={
            "prompt": prompt,
        },
    )

    print(f"  ⛓️  QA Chain built (chain_type=stuff, return_sources=True)")
    return qa_chain


# ╔════════════════════════════════════════════════════════════════╗
# ║               6. EXPLAINABILITY & OUTPUT                      ║
# ╚════════════════════════════════════════════════════════════════╝

def print_results(query: str, result: dict):
    """
    Print the generated answer AND the exact metadata of every
    source document used to generate it.
    
    This provides full explainability and traceability back to the
    original 3GPP specifications.
    """
    print(f"\n{'═'*60}")
    print(f"  📝 QUESTION")
    print(f"{'═'*60}")
    print(f"  {query}")

    print(f"\n{'═'*60}")
    print(f"  💡 ANSWER")
    print(f"{'═'*60}")
    print(f"\n{result['result']}")

    print(f"\n{'═'*60}")
    print(f"  📚 SOURCE DOCUMENTS (Explainability)")
    print(f"{'═'*60}")

    source_docs = result.get("source_documents", [])
    if not source_docs:
        print("  ⚠️  No source documents returned.")
        return

    seen = set()
    for i, doc in enumerate(source_docs, 1):
        meta = doc.metadata
        # Deduplicate by (spec_number, doc_id)
        dedup_key = (meta.get("spec_number", ""), meta.get("doc_id", ""))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        print(f"\n  ┌─ Source {i} {'─'*45}")
        print(f"  │ Spec Number : TS {meta.get('spec_number', 'N/A')}")
        print(f"  │ Document ID : {meta.get('doc_id', 'N/A')}")
        print(f"  │ Series      : {meta.get('series', 'N/A')}")
        print(f"  │ Release     : {meta.get('release', 'N/A')}")
        print(f"  │ Filename    : {meta.get('filename', 'N/A')}")
        print(f"  │ Source File : {meta.get('source_file', 'N/A')}")
        # Show a preview of the chunk content
        preview = doc.page_content[:200].replace("\n", " ").strip()
        print(f"  │ Content     : {preview}…")
        print(f"  └{'─'*55}")

    print()


# ╔════════════════════════════════════════════════════════════════╗
# ║                    7. MAIN PIPELINE                           ║
# ╚════════════════════════════════════════════════════════════════╝

def validate_api_key() -> str:
    """Validate that the Google API key is available."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("\n  ❌ ERROR: GOOGLE_API_KEY not found!")
        print("     Set it in a .env file or as an environment variable:")
        print('     export GOOGLE_API_KEY="your-api-key-here"')
        print('     Or create a .env file with: GOOGLE_API_KEY=your-api-key-here')
        sys.exit(1)
    return api_key


def main():
    """
    Main entry point — orchestrates the full RAG pipeline:
      1. Validate API key
      2. Initialize embeddings
      3. Build or load ChromaDB vector store
      4. Create MMR retriever
      5. Initialize Gemini LLM
      6. Build QA chain
      7. Interactive query loop
    """
    print("\n" + "═" * 60)
    print("  🚀 3GPP RAG Assistant — Initializing…")
    print("═" * 60)

    # ── Step 1: API Key ──────────────────────────────────────
    api_key = validate_api_key()
    print(f"  ✅ Google API key loaded")

    # ── Step 2: Embeddings ───────────────────────────────────
    embeddings = get_embedding_model()

    # ── Step 3: Vector Store ─────────────────────────────────
    vectorstore = build_or_load_vectorstore(embeddings, JSON_DATA_DIR)

    # ── Step 4: Retriever ────────────────────────────────────
    retriever = create_retriever(vectorstore)

    # ── Step 5: LLM ──────────────────────────────────────────
    print(f"\n  🤖 Initializing Gemini LLM: {GEMINI_MODEL}")
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_TOKENS,
        convert_system_message_to_human=True,
        max_retries=5,  # Added to handle temporary 429 quota errors
    )
    print(f"  ✅ Gemini LLM ready")

    # ── Step 6: QA Chain ─────────────────────────────────────
    qa_chain = build_qa_chain(retriever, llm)

    # ── Step 7: Interactive Query Loop ───────────────────────
    print(f"\n{'═'*60}")
    print(f"  ✅ 3GPP RAG Assistant is READY!")
    print(f"  💬 Ask questions about 3GPP Release 16 & 18 specs.")
    print(f"  ⏎  Press Enter TWICE to submit a multi-line question.")
    print(f"  ⏎  Type 'quit' or 'exit' and press Enter to stop.")
    print(f"{'═'*60}\n")

    while True:
        try:
            print("  🔎 Your question: ")
            lines = []
            while True:
                line = input("     ")
                # If they type quit on the first line, exit
                if not lines and line.strip().lower() in ("quit", "exit", "q"):
                    print("\n  👋 Goodbye!")
                    return
                # If they hit enter on an empty line and we have text, submit it
                if not line.strip() and lines:
                    break
                lines.append(line)
                
            query = "\n".join(lines).strip()
            
        except (EOFError, KeyboardInterrupt):
            print("\n\n  👋 Goodbye!")
            break

        if not query:
            continue

        print(f"\n  ⏳ Retrieving & generating answer…\n")

        try:
            start_time = time.time()
            result = qa_chain.invoke({"query": query})
            elapsed = time.time() - start_time

            print_results(query, result)
            print(f"  ⏱️  Response time: {elapsed:.2f}s\n")

        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            if "429" in str(e):
                print(f"     ⚠️  Rate limit reached. Please wait a minute before asking another question.")
            else:
                print(f"     Please try rephrasing your question.\n")


# ── Script Entry Point ───────────────────────────────────────────
if __name__ == "__main__":
    main()
