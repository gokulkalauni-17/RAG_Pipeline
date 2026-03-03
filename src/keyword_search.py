import re
from typing import List, Dict, Any
from collections import Counter

class KeywordSearch:
    """Simple keyword-based search using TF-IDF and BM25 concepts"""
    
    def __init__(self):
        self.documents = []
        self.inverted_index = {}
        self.doc_lengths = []
        
    def build_index(self, documents: List[Dict[str, Any]]):
        """Build inverted index from documents"""
        self.documents = documents
        self.inverted_index = {}
        self.doc_lengths = []
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get("metadata", {}).get("text", "")
            # Tokenize and normalize
            tokens = self._tokenize(text)
            self.doc_lengths.append(len(tokens))
            
            # Update inverted index
            for token in set(tokens):
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append(doc_idx)
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization and normalization"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Split into tokens
        tokens = text.split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'was', 'be', 'been'}
        return [t for t in tokens if t and t not in stop_words]
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents by keywords"""
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            return []
        
        # Calculate relevance scores
        scores = {}
        for token in query_tokens:
            if token in self.inverted_index:
                for doc_idx in self.inverted_index[token]:
                    scores[doc_idx] = scores.get(doc_idx, 0) + 1
        
        # Sort by relevance and return top k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        
        for doc_idx, score in sorted_docs:
            doc = self.documents[doc_idx]
            results.append({
                "index": doc_idx,
                "score": score,
                "metadata": doc.get("metadata", {}),
                "search_type": "keyword"
            })
        
        return results
    
    def bm25_search(self, query: str, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
        """BM25 search algorithm - more sophisticated than simple keyword matching"""
        query_tokens = self._tokenize(query)
        
        if not query_tokens or not self.documents:
            return []
        
        N = len(self.documents)  # Total documents
        avg_doc_len = sum(self.doc_lengths) / N if N > 0 else 0
        
        scores = {}
        
        for token in query_tokens:
            if token in self.inverted_index:
                # IDF calculation
                df = len(self.inverted_index[token])
                idf = max(0, (N - df + 0.5) / (df + 0.5))
                
                # BM25 score for each document
                for doc_idx in self.inverted_index[token]:
                    # Count occurrences of token in document
                    text = self.documents[doc_idx].get("metadata", {}).get("text", "")
                    tokens_in_doc = self._tokenize(text)
                    tf = tokens_in_doc.count(token)
                    
                    # BM25 formula
                    doc_len = self.doc_lengths[doc_idx]
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * (doc_len / avg_doc_len))
                    
                    bm25_score = idf * (numerator / denominator)
                    scores[doc_idx] = scores.get(doc_idx, 0) + bm25_score
        
        # Sort and return top k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        
        for doc_idx, score in sorted_docs:
            doc = self.documents[doc_idx]
            results.append({
                "index": doc_idx,
                "score": float(score),
                "metadata": doc.get("metadata", {}),
                "search_type": "bm25"
            })
        
        return results
