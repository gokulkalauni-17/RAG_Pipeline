import numpy as np
from typing import List, Dict, Any, Optional
from src.vectorstore import FaissVectorStore
from src.keyword_search import KeywordSearch
from src.knowledge_graph import SimpleKnowledgeGraph

class HybridSearch:
    """Combine semantic search, keyword search, and knowledge graph for better retrieval"""
    
    def __init__(self, vector_store: FaissVectorStore, keyword_search: KeywordSearch, 
                 knowledge_graph: SimpleKnowledgeGraph):
        self.vector_store = vector_store
        self.keyword_search = keyword_search
        self.knowledge_graph = knowledge_graph
        self.weights = {
            "semantic": 0.5,  # Vector similarity weight
            "keyword": 0.3,   # Keyword search weight
            "kg": 0.2         # Knowledge graph weight
        }
    
    def set_weights(self, semantic: float = 0.5, keyword: float = 0.3, kg: float = 0.2):
        """Adjust search method weights (must sum to 1.0)"""
        total = semantic + keyword + kg
        if total <= 0:
            raise ValueError("Weights must sum to positive value")
        
        self.weights["semantic"] = semantic / total
        self.weights["keyword"] = keyword / total
        self.weights["kg"] = kg / total
        print(f"[INFO] Updated search weights - Semantic: {self.weights['semantic']:.2f}, Keyword: {self.weights['keyword']:.2f}, KG: {self.weights['kg']:.2f}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining all three methods"""
        
        # 1. Semantic search (vector similarity)
        semantic_results = self.vector_store.query(query, top_k=top_k)
        semantic_scores = self._normalize_scores(
            [(r["index"], 1 - r["distance"] / 2) for r in semantic_results if "index" in r]
        )
        
        # 2. Keyword search (TF-IDF/BM25)
        keyword_results = self.keyword_search.bm25_search(query, top_k=top_k)
        keyword_scores = self._normalize_scores(
            [(r["index"], r["score"]) for r in keyword_results]
        )
        
        # 3. Knowledge graph search (entity-based)
        kg_results = self._kg_search(query, top_k=top_k)
        kg_scores = self._normalize_scores(
            [(r["index"], r["score"]) for r in kg_results]
        )
        
        # Combine scores with weights
        combined_scores = self._combine_scores(semantic_scores, keyword_scores, kg_scores)
        
        # Format results
        results = self._format_results(combined_scores, top_k)
        
        return results
    
    def _normalize_scores(self, scores: List[tuple]) -> Dict[int, float]:
        """Normalize scores to 0-1 range"""
        if not scores:
            return {}
        
        scores_dict = dict(scores)
        values = list(scores_dict.values())
        
        if not values:
            return scores_dict
        
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {k: 0.5 for k in scores_dict.keys()}
        
        return {
            k: (v - min_val) / (max_val - min_val) 
            for k, v in scores_dict.items()
        }
    
    def _kg_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using knowledge graph"""
        results = []
        
        # Extract entities from query
        entities = self.knowledge_graph._extract_entities(query)
        
        if not entities:
            return results
        
        # Find documents mentioning these entities
        doc_scores = {}
        for entity in entities:
            matching_entities = self.knowledge_graph.search_by_entity(entity)
            
            for match_entity in matching_entities:
                docs = self.knowledge_graph.get_entity_documents(match_entity)
                for doc_idx in docs:
                    doc_scores[doc_idx] = doc_scores.get(doc_idx, 0) + 1
        
        # Sort and return top k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for doc_idx, score in sorted_docs:
            doc = self.vector_store.metadata[doc_idx] if doc_idx < len(self.vector_store.metadata) else {}
            results.append({
                "index": doc_idx,
                "score": float(score),
                "metadata": doc,
                "search_type": "knowledge_graph"
            })
        
        return results
    
    def _combine_scores(self, semantic_scores: Dict[int, float], 
                       keyword_scores: Dict[int, float],
                       kg_scores: Dict[int, float]) -> Dict[int, float]:
        """Combine scores from all three search methods"""
        combined = {}
        all_indices = set(semantic_scores.keys()) | set(keyword_scores.keys()) | set(kg_scores.keys())
        
        for idx in all_indices:
            sem_score = semantic_scores.get(idx, 0) * self.weights["semantic"]
            key_score = keyword_scores.get(idx, 0) * self.weights["keyword"]
            kg_score = kg_scores.get(idx, 0) * self.weights["kg"]
            
            combined[idx] = sem_score + key_score + kg_score
        
        return combined
    
    def _format_results(self, scores: Dict[int, float], top_k: int) -> List[Dict[str, Any]]:
        """Format final results"""
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        results = []
        for doc_idx, combined_score in sorted_scores:
            doc = self.vector_store.metadata[doc_idx] if doc_idx < len(self.vector_store.metadata) else {}
            results.append({
                "index": doc_idx,
                "combined_score": float(combined_score),
                "metadata": doc,
                "search_type": "hybrid"
            })
        
        return results
    
    def search_with_explanation(self, query: str, top_k: int = 5) -> tuple:
        """Search and return detailed explanation of scoring"""
        
        # Get individual results
        semantic_results = self.vector_store.query(query, top_k=top_k)
        keyword_results = self.keyword_search.bm25_search(query, top_k=top_k)
        kg_results = self._kg_search(query, top_k=top_k)
        
        # Get final results
        final_results = self.search(query, top_k=top_k)
        
        # Create explanation
        explanation = {
            "query": query,
            "semantic_results": semantic_results,
            "keyword_results": keyword_results,
            "kg_results": kg_results,
            "final_results": final_results,
            "weights": self.weights
        }
        
        return final_results, explanation
