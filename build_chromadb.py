"""
╔══════════════════════════════════════════════════════════════════╗
║        ChromaDB Builder — Intel Arc GPU Accelerated             ║
║──────────────────────────────────────────────────────────────────║
║  Builds the persistent ChromaDB vector store from 3GPP JSON     ║
║  specs using OpenVINO on Intel Arc GPU for fast embedding.      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import glob
import time

# ── Environment settings ─────────────────────────────────────────
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ── Configuration ────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 5000  # Larger batches since GPU is fast

# JSON field mapping (adjust if your schema changes)
CONTENT_KEY = "text"
METADATA_KEYS = {
    "id": "doc_id",
    "spec_num": "spec_number",
    "series": "series",
    "filename": "filename",
    "num_paragraphs": "num_paragraphs",
    "num_chars": "num_chars",
}


# ╔════════════════════════════════════════════════════════════════╗
# ║            1. DETECT BEST AVAILABLE DEVICE                    ║
# ╚════════════════════════════════════════════════════════════════╝

def detect_device():
    """
    Detect the best available compute device for embeddings.
    Priority: GPU (Intel Arc) > NPU > CPU
    """
    try:
        from openvino import Core
        core = Core()
        devices = core.available_devices
        print(f"  OpenVINO devices detected: {devices}")

        if "GPU" in devices:
            print("  >> Using Intel Arc GPU for embeddings")
            return "GPU"
        elif "NPU" in devices:
            print("  >> Using Intel NPU for embeddings")
            return "NPU"
    except ImportError:
        print("  >> OpenVINO not available")

    print("  >> Falling back to CPU")
    return "CPU"


# ╔════════════════════════════════════════════════════════════════╗
# ║           2. CREATE OPENVINO-ACCELERATED EMBEDDINGS           ║
# ╚════════════════════════════════════════════════════════════════╝

def create_embeddings(device: str):
    """
    Create embeddings using either OpenVINO (GPU/NPU) or HuggingFace (CPU).
    
    OpenVINO converts the model to its intermediate representation (IR)
    and runs it optimized on Intel hardware, including the Arc GPU.
    """
    if device in ("GPU", "NPU"):
        try:
            from optimum.intel import OVModelForFeatureExtraction
            from transformers import AutoTokenizer
            import numpy as np

            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            print(f"\n  Loading OpenVINO model on {device}...")
            print(f"  Model: {model_name}")
            print(f"  (First run will convert model to OpenVINO IR format)")

            # Export/load the model in OpenVINO format
            ov_model = OVModelForFeatureExtraction.from_pretrained(
                model_name,
                export=True,
                device=device,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            print(f"  ✅ OpenVINO model loaded on {device}")

            # Return a wrapper that matches LangChain's embedding interface
            return OpenVINOEmbeddings(ov_model, tokenizer, device)

        except Exception as e:
            print(f"  ⚠️ OpenVINO on {device} failed: {e}")
            print(f"  Falling back to CPU...")
            device = "CPU"

    # CPU fallback
    from langchain_huggingface import HuggingFaceEmbeddings
    print(f"\n  Loading HuggingFace model on CPU...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 256},
    )
    print(f"  ✅ HuggingFace embedding model ready (CPU)")
    return embeddings


class OpenVINOEmbeddings:
    """
    LangChain-compatible wrapper around an OpenVINO model.
    Implements embed_documents() and embed_query() for ChromaDB.
    """
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _mean_pooling(self, model_output, attention_mask):
        """Apply mean pooling to get sentence embeddings."""
        import numpy as np
        import torch

        token_embeddings = model_output[0]
        if isinstance(token_embeddings, torch.Tensor):
            token_embeddings = token_embeddings.detach().numpy()

        mask_expanded = attention_mask.numpy().astype(float)
        mask_expanded = np.expand_dims(mask_expanded, axis=-1)
        mask_expanded = np.broadcast_to(mask_expanded, token_embeddings.shape)

        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        embeddings = sum_embeddings / sum_mask

        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, a_min=1e-9, a_max=None)
        embeddings = embeddings / norms

        return embeddings.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents in batches."""
        all_embeddings = []
        batch_size = 128

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            outputs = self.model(**encoded)
            embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self.embed_documents([text])[0]


# ╔════════════════════════════════════════════════════════════════╗
# ║                  3. JSON INGESTION                            ║
# ╚════════════════════════════════════════════════════════════════╝

def detect_release(filename: str, text: str) -> str:
    """Detect 3GPP release number from filename or text."""
    fname_lower = filename.lower()
    if "16" in fname_lower:
        return "16"
    elif "18" in fname_lower:
        return "18"
    snippet = text[:500].lower()
    if "release 18" in snippet:
        return "18"
    elif "release 16" in snippet:
        return "16"
    elif "release 17" in snippet:
        return "17"
    return "Unknown"


def load_and_chunk() -> list[Document]:
    """Load all 3GPP JSON files and chunk them."""
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "3gpp*.json")))
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.json")))

    print(f"\n  Found {len(json_files)} spec file(s)")

    all_docs = []
    for json_path in json_files:
        fname = os.path.basename(json_path)
        print(f"\n  Loading: {fname}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        entries = data if isinstance(data, list) else list(data.values())
        print(f"    {len(entries):,} entries")

        for entry in tqdm(entries, desc=f"    Parsing", unit="doc"):
            content = entry.get(CONTENT_KEY, "")
            if not content or not content.strip():
                continue

            metadata = {}
            for json_key, meta_name in METADATA_KEYS.items():
                metadata[meta_name] = entry.get(json_key, "N/A")
            metadata["release"] = detect_release(fname, content)
            metadata["source_file"] = fname

            all_docs.append(Document(page_content=content, metadata=metadata))

    print(f"\n  Total documents: {len(all_docs):,}")

    # Chunk
    print(f"\n  Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(all_docs)
    print(f"  Created {len(chunks):,} chunks")
    return chunks


# ╔════════════════════════════════════════════════════════════════╗
# ║                     4. BUILD CHROMADB                         ║
# ╚════════════════════════════════════════════════════════════════╝

def build_chromadb():
    """Main function to build ChromaDB with GPU acceleration."""
    print("\n" + "=" * 60)
    print("  ChromaDB Builder — Intel Arc GPU Accelerated")
    print("=" * 60)

    # Check if already exists
    if os.path.isdir(CHROMA_PERSIST_DIR) and os.listdir(CHROMA_PERSIST_DIR):
        print(f"\n  ChromaDB already exists at: {CHROMA_PERSIST_DIR}")
        response = input("  Delete and rebuild? (y/n): ").strip().lower()
        if response != "y":
            print("  Keeping existing database. Exiting.")
            return
        import shutil
        shutil.rmtree(CHROMA_PERSIST_DIR)
        print("  Deleted old database.")

    # Detect device
    print(f"\n{'─'*60}")
    print("  Detecting compute device...")
    print(f"{'─'*60}")
    device = detect_device()

    # Create embeddings
    print(f"\n{'─'*60}")
    print("  Initializing embedding model...")
    print(f"{'─'*60}")
    embeddings = create_embeddings(device)

    # Quick sanity check
    test_emb = embeddings.embed_query("What is 5G?")
    print(f"  Embedding dimension: {len(test_emb)}")

    # Load and chunk documents
    print(f"\n{'─'*60}")
    print("  Loading & chunking 3GPP specifications...")
    print(f"{'─'*60}")
    chunks = load_and_chunk()

    # Build ChromaDB in batches
    print(f"\n{'─'*60}")
    print(f"  Embedding {len(chunks):,} chunks into ChromaDB...")
    print(f"  Device: {device} | Batch size: {BATCH_SIZE}")
    print(f"{'─'*60}")

    vectorstore = None
    start_time = time.time()
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(chunks), BATCH_SIZE),
                  desc="  Building", unit="batch", total=total_batches):
        batch = chunks[i : i + BATCH_SIZE]

        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_PERSIST_DIR,
                collection_name="3gpp_specs",
            )
        else:
            vectorstore.add_documents(documents=batch)

    elapsed = time.time() - start_time

    # Final stats
    count = vectorstore._collection.count()
    print(f"\n{'=' * 60}")
    print(f"  ✅ ChromaDB built successfully!")
    print(f"  ✅ Vectors stored : {count:,}")
    print(f"  ✅ Persisted to   : {CHROMA_PERSIST_DIR}")
    print(f"  ✅ Time taken     : {elapsed/60:.1f} minutes")
    print(f"  ✅ Device used    : {device}")
    print(f"{'=' * 60}")

    # Quick retrieval test
    print(f"\n  Quick retrieval test...")
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.7},
    )
    results = retriever.invoke("What is 5G NR?")
    for j, doc in enumerate(results, 1):
        m = doc.metadata
        preview = doc.page_content[:100].replace("\n", " ")
        print(f"  [{j}] TS {m.get('spec_number')} (R{m.get('release')}) — {preview}...")

    print(f"\n  Done! You can now run main.py to query the database.")


if __name__ == "__main__":
    build_chromadb()
