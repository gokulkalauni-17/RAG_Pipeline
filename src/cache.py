import os
from typing import Optional, Tuple
from src.semantic_cache_faiss import SemanticCacheFAISS

# Initialize semantic cache with FAISS backend
_semantic_cache = SemanticCacheFAISS(
    cache_dir="faiss_store/semantic_cache",
    distance_threshold=0.30  # L2 distance threshold for semantic similarity
)



def get_cached_result_semantic(query: str) -> Optional[Tuple[str, float]]:
    """
    Get cached result using semantic cache with FAISS.
    Returns (result, cosine_similarity) or None if no match found.
    """
    cache_result = _semantic_cache.check(query, num_results=1)
    
    if cache_result.hit and cache_result.best_match:
        return (cache_result.best_match.response, cache_result.best_match.cosine_similarity)
    
    return None

def get_cached_result_with_similarity(query: str) -> Optional[Tuple[str, float]]:
    """Get cached result with similarity score using FAISS semantic cache"""
    return get_cached_result_semantic(query)

def cache_result(query: str, result: str):
    """Cache a query result with semantic indexing via FAISS"""
    _semantic_cache.add_pair(query, result)

def clear_cache():
    """Clear all cached queries"""
    _semantic_cache.clear()

def get_cache_stats() -> dict:
    """Get cache statistics"""
    semantic_stats = _semantic_cache.get_stats()
    return {
        "semantic_cache_entries": semantic_stats["total_entries"],
        "semantic_distance_threshold": semantic_stats["distance_threshold"]
    }

def set_similarity_threshold(threshold: float):
    """Update similarity threshold for FAISS cache (0.0 to 1.0)"""
    if 0.0 <= threshold <= 1.0:
        _semantic_cache.set_distance_threshold(1.0 - threshold)  # Convert similarity to L2 distance
    else:
        print("[ERROR] Threshold must be between 0.0 and 1.0")

# Example usage
if __name__ == "__main__":
    # Test semantic caching with FAISS
    test_query1 = "What is machine learning?"
    test_query2 = "What does machine learning mean?"  # Similar query
    test_query3 = "How do I cook pasta?"  # Different query
    
    # First call - should be None
    result = get_cached_result_semantic(test_query1)
    print(f"First call result: {result}")
    
    # Cache a result
    cache_result(test_query1, "Machine learning is a subset of AI...")
    
    # Second call with similar query - should return cached result
    result = get_cached_result_semantic(test_query2)
    print(f"Similar query result: {result}")
    
    # Third call with different query - should return None
    result = get_cached_result_semantic(test_query3)
    print(f"Different query result: {result}")
    
    # Get stats
    stats = get_cache_stats()
    print(f"Cache stats: {stats}")
