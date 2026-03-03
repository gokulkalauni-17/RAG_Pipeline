import os
import faiss
import numpy as np
import json
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline
from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_WORKERS, DEVICE

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any], force: bool = False):
        """Build or update the vector store from raw documents.
        If an index already exists and `force` is False the method will skip rebuilding.
        """
        # if persistence exists and not forcing a rebuild, just load existing
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        if os.path.exists(faiss_path) and not force:
            print(f"[INFO] Existing index found at {faiss_path}; skipping rebuild.")
            return
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, device=DEVICE)
        chunks = emb_pipe.chunk_documents(documents)
        # Use parallel embedding with configured batch_size and max_workers from config
        embeddings = emb_pipe.embed_chunks(chunks, batch_size=EMBEDDING_BATCH_SIZE, max_workers=EMBEDDING_MAX_WORKERS)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        # print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.json")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        # print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.json")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        # print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        # print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    # build (skips if index already exists)
    store.build_from_documents(docs)
    # force rebuild if you changed configuration or want a fresh index
    # store.build_from_documents(docs, force=True)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
