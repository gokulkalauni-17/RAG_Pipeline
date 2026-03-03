"""
Semantic Cache using FAISS and SentenceTransformers.
Enables semantic similarity matching for query results caching.
"""

import json
import os
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import faiss
from datetime import datetime


@dataclass
class CacheEntry:
    """Represents a cached Q&A pair."""
    query: str
    response: str
    embedding: List[float]
    distance: float = 0.0
    cosine_similarity: float = 0.0
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class CacheCheckResult:
    """Result from cache lookup."""
    hit: bool
    best_match: Optional[CacheEntry] = None
    all_matches: List[CacheEntry] = None
    
    def __post_init__(self):
        if self.all_matches is None:
            self.all_matches = []


class SemanticCacheFAISS:
    """Semantic cache using FAISS for efficient similarity search."""
    
    def __init__(
        self,
        cache_dir: str = "faiss_store/semantic_cache",
        embedding_model: str = "all-MiniLM-L6-v2",
        distance_threshold: float = 0.1
    ):
        """
        Initialize semantic cache with FAISS backend.
        
        Args:
            cache_dir: Directory to persist cache files
            embedding_model: SentenceTransformer model to use
            distance_threshold: Maximum L2 distance for cache hit (lower = stricter)
        """
        self.cache_dir = cache_dir
        self.distance_threshold = distance_threshold
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.entries: List[CacheEntry] = []
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache
        self._load_cache()
       
    def set_distance_threshold(self, threshold: float) -> None:
        """Update the L2 distance threshold for cache hits."""
        if threshold < 0:
            print("[ERROR] Distance threshold must be non-negative")
            return
        self.distance_threshold = threshold
        print(f"[INFO] Distance threshold updated to {threshold}")
    
    def add_pair(self, query: str, response: str) -> None:
        """Add a Q&A pair to the cache."""
        embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        entry = CacheEntry(
            query=query,
            response=response,
            embedding=embedding.tolist()
        )
        
        self.entries.append(entry)
        self.index.add(embedding.reshape(1, -1).astype('float32'))
        
        self._save_cache()
    
    def check(self, query: str, num_results: int = 1) -> CacheCheckResult:
        """
        Check cache for similar queries.
        
        Args:
            query: Query to check
            num_results: Number of top matches to return
            
        Returns:
            CacheCheckResult with hit status and matches
        """
        if len(self.entries) == 0:
            return CacheCheckResult(hit=False)
        
        # Encode query
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        
        # Search FAISS index
        k = min(num_results, len(self.entries))
        distances, indices = self.index.search(
            query_embedding.reshape(1, -1).astype('float32'), 
            k
        )
        
        matches = []
        for idx, distance in zip(indices[0], distances[0]):
            entry = self.entries[idx]
            entry.distance = float(distance)
            
            # Calculate cosine similarity from embeddings
            cached_emb = np.array(entry.embedding)
            query_norm = np.linalg.norm(query_embedding)
            cached_norm = np.linalg.norm(cached_emb)
            
            if query_norm > 0 and cached_norm > 0:
                cosine_sim = np.dot(query_embedding, cached_emb) / (query_norm * cached_norm)
                entry.cosine_similarity = float(np.clip(cosine_sim, -1.0, 1.0))
            else:
                entry.cosine_similarity = 0.0
            
            matches.append(entry)
        
        # Check if best match meets threshold
        best_match = matches[0] if matches else None
        is_hit = best_match is not None and best_match.distance <= self.distance_threshold
        
        return CacheCheckResult(
            hit=is_hit,
            best_match=best_match if is_hit else None,
            all_matches=matches
        )
    
    def hydrate_from_pairs(self, pairs: List[Tuple[str, str]]) -> None:
        """Load Q&A pairs into cache."""
        for question, answer in pairs:
            self.add_pair(question, answer)
    
    def _save_cache(self) -> None:
        """Persist cache to disk."""
        try:
            # Save FAISS index
            faiss.write_index(
                self.index,
                os.path.join(self.cache_dir, "cache.index")
            )
            
            # Save cache entries metadata
            entries_data = [
                {
                    "query": entry.query,
                    "response": entry.response,
                    "embedding": entry.embedding,
                    "timestamp": entry.timestamp
                }
                for entry in self.entries
            ]
            
            with open(os.path.join(self.cache_dir, "cache_entries.json"), 'w') as f:
                json.dump(entries_data, f, indent=2)
                
        except Exception as e:
            print(f"[ERROR] Failed to save semantic cache: {e}")
    
    def _load_cache(self) -> None:
        """Load cache from disk."""
        try:
            index_path = os.path.join(self.cache_dir, "cache.index")
            entries_path = os.path.join(self.cache_dir, "cache_entries.json")
            
            if os.path.exists(index_path) and os.path.exists(entries_path):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load entries metadata
                with open(entries_path, 'r') as f:
                    entries_data = json.load(f)
                
                self.entries = [
                    CacheEntry(
                        query=e["query"],
                        response=e["response"],
                        embedding=e["embedding"],
                        timestamp=e.get("timestamp")
                    )
                    for e in entries_data
                ]
                
                print(f"[INFO] Loaded semantic cache with {len(self.entries)} entries")
                
        except Exception as e:
            print(f"[WARN] Failed to load semantic cache: {e}")
    
    def save_to_file(self, filepath: str) -> None:
        """Export cache entries to CSV."""
        import csv
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['question', 'answer', 'timestamp'])
                for entry in self.entries:
                    writer.writerow([entry.query, entry.response, entry.timestamp])
            print(f"[INFO] Cache saved to {filepath}")
        except Exception as e:
            print(f"[ERROR] Failed to save cache to file: {e}")
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "total_entries": len(self.entries),
            "distance_threshold": self.distance_threshold,
            "embedding_model": "all-MiniLM-L6-v2",
            "cache_dir": self.cache_dir
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.entries = []
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self._save_cache()
        print("[INFO] Cache cleared")
