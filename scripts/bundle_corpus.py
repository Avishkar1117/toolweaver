"""Build the minimal Chroma store that gets baked into the deploy image.

doc_lookup needs only the ``doc_eval_corpus`` collection (90 chunks) from P2's
store, but that store also holds another collection plus stale segment dirs
(~112 MB). HF Spaces has no runtime volume mounts, so we bake the corpus into the
image -- and ship only what's needed. This re-creates just ``doc_eval_corpus``
into a fresh ~15 MB store, preserving the exact Gemini vectors and the default L2
space (the source collection sets no custom ``hnsw:space``).

Run once before deploying:  ``python scripts/bundle_corpus.py``
"""

import shutil
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

NAME = "doc_eval_corpus"
SRC = Path(r"C:/Users/avish/Desktop/rag-service/chroma_store")
DST = Path(__file__).resolve().parents[1] / "chroma_store"


def main() -> None:
    cs = ChromaSettings(anonymized_telemetry=False)
    src = chromadb.PersistentClient(path=str(SRC), settings=cs).get_collection(NAME)
    data = src.get(include=["embeddings", "documents", "metadatas"])

    if DST.exists():
        shutil.rmtree(DST)
    dst = chromadb.PersistentClient(path=str(DST), settings=cs).create_collection(NAME)
    dst.add(
        ids=data["ids"],
        embeddings=data["embeddings"],
        documents=data["documents"],
        metadatas=data["metadatas"],
    )

    print("rebuilt", NAME, "chunks:", dst.count())
    size = sum(f.stat().st_size for f in DST.rglob("*") if f.is_file())
    print(f"bundled store size: {size / 1e6:.1f} MB at {DST}")


if __name__ == "__main__":
    main()
