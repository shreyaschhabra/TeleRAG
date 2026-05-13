"""Quick end-to-end pipeline test (no LLM / no API key needed)."""
import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(__file__))
from main import load_json_documents, chunk_documents, get_embedding_model, build_or_load_vectorstore, create_retriever

print("=" * 60)
print("  FULL PIPELINE TEST (without LLM)")
print("=" * 60)

# Step 1: Embeddings
embeddings = get_embedding_model()

# Step 2: Build/Load vector store
data_dir = os.path.dirname(os.path.abspath(__file__))
vectorstore = build_or_load_vectorstore(embeddings, data_dir)

# Step 3: Create retriever
retriever = create_retriever(vectorstore)

# Step 4: Test a retrieval query
print("\n" + "=" * 60)
print("  TEST RETRIEVAL: What is 5G NR?")
print("=" * 60)
docs = retriever.invoke("What is 5G NR?")
for i, d in enumerate(docs[:3], 1):
    m = d.metadata
    print(f"\n  [{i}] Spec: TS {m.get('spec_number')}, Release: {m.get('release')}, File: {m.get('filename')}")
    preview = d.page_content[:120].replace("\n", " ")
    print(f"      Preview: {preview}...")

print("\n" + "=" * 60)
print("  PIPELINE READY - Just needs GOOGLE_API_KEY to go live!")
print("=" * 60)
