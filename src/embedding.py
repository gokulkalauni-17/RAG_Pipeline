from typing import List, Any
from concurrent.futures import ThreadPoolExecutor
import psutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents
from src.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MAX_WORKERS, DEVICE

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200, device: str = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # allow overriding device or fall back to config/auto
        if device is None:
            device = DEVICE
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception:
            print(f"[WARN] Unable to load model on device {device}, falling back to cpu")
            self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name} on {device}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def _encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode a batch of texts."""
        return self.model.encode(texts, show_progress_bar=False)

    def embed_chunks(self, chunks: List[Any], batch_size: int = None, max_workers: int = None) -> np.ndarray:
        """
        Generate embeddings for chunks using parallel batch processing.
        
        Args:
            chunks: List of document chunks to embed
            batch_size: Number of texts per batch (default: EMBEDDING_BATCH_SIZE)
            max_workers: Number of parallel batch workers (default: EMBEDDING_MAX_WORKERS)
        
        Returns:
            numpy array of embeddings
        """
        if batch_size is None:
            batch_size = EMBEDDING_BATCH_SIZE
        if max_workers is None:
            max_workers = EMBEDDING_MAX_WORKERS
        # dynamically reduce batch size if memory is low
        vm = psutil.virtual_memory()
        if vm.available < 2 * 1024 ** 3:  # less than 2GB free
            orig = batch_size
            batch_size = max(8, batch_size // 2)
            print(f"[WARN] Low memory ({vm.available // (1024**2)}MB); reducing batch size from {orig} to {batch_size}")
        
        texts = [chunk.page_content for chunk in chunks]
        total_texts = len(texts)
        print(f"[INFO] Generating embeddings for {total_texts} chunks in batches of {batch_size} using {max_workers} workers...")
        
        # Create batches
        batches = [texts[i:i + batch_size] for i in range(0, total_texts, batch_size)]
        num_batches = len(batches)
        print(f"[INFO] Processing {num_batches} batches...")
        
        all_embeddings = []
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._encode_batch, batch) for batch in batches]
            
            for idx, future in enumerate(futures, 1):
                try:
                    batch_embeddings = future.result()
                    all_embeddings.extend(batch_embeddings)
                    print(f"[INFO] Processed batch {idx}/{num_batches}")
                except Exception as e:
                    print(f"[ERROR] Error processing batch {idx}: {e}")
                    raise
        
        embeddings_array = np.array(all_embeddings)
        print(f"[INFO] Embeddings shape: {embeddings_array.shape}")
        return embeddings_array

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline(device='cuda' if psutil.virtual_memory().total > 8 * 1024**3 else 'cpu')
    chunks = emb_pipe.chunk_documents(docs)
    # Use default parallel embedding settings from config
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
    
    # Example with custom settings
    embeddings_custom = emb_pipe.embed_chunks(chunks, batch_size=64, max_workers=3)
    print("[INFO] Custom batch embedding completed.")
