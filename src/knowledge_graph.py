import json
import os
from typing import List, Dict, Any, Set, Tuple
from collections import defaultdict
import re

class SimpleKnowledgeGraph:
    """Simple in-memory knowledge graph for entity relationships"""
    
    def __init__(self, persist_path: str = "faiss_store/knowledge_graph.json"):
        self.persist_path = persist_path
        self.entities: Set[str] = set()
        self.relationships: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)  # entity -> [(relation, target_entity, doc_source)]
        self.entity_documents: Dict[str, Set[int]] = defaultdict(set)  # entity -> set of document indices
        self.load()
    
    def build_from_documents(self, documents: List[Dict[str, Any]]):
        """Extract entities and relationships from documents"""
        print("[INFO] Building knowledge graph from documents...")
        
        for doc_idx, doc in enumerate(documents):
            text = doc.get("metadata", {}).get("text", "")
            if not text:
                continue
            
            # Extract entities (simple pattern matching)
            entities = self._extract_entities(text)
            for entity in entities:
                self.entities.add(entity)
                self.entity_documents[entity].add(doc_idx)
            
            # Extract relationships (simple patterns)
            relationships = self._extract_relationships(text, entities)
            for source_entity, relation, target_entity in relationships:
                self.relationships[source_entity].append((relation, target_entity, str(doc_idx)))
        
        print(f"[INFO] Knowledge graph built: {len(self.entities)} entities, {len(self.relationships)} relationships")
        self.save()
    
    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entities from text (capitalized words, technical terms)"""
        entities = set()
        
        # Find capitalized phrases (proper nouns)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities.update(capitalized)
        
        # Find common technical terms
        technical_terms = [
            'algorithm', 'model', 'network', 'layer', 'training', 
            'optimization', 'gradient', 'loss', 'accuracy', 'embedding',
            'attention', 'transformer', 'encoder', 'decoder', 'vector',
            'matrix', 'tensor', 'parameter', 'weight', 'bias'
        ]
        
        text_lower = text.lower()
        for term in technical_terms:
            if term in text_lower:
                # Find the context (capitalized version if exists)
                pattern = re.compile(r'\b' + term + r'\b', re.IGNORECASE)
                matches = pattern.findall(text)
                if matches:
                    entities.add(term.title())
        
        return entities
    
    def _extract_relationships(self, text: str, entities: Set[str]) -> List[Tuple[str, str, str]]:
        """Extract relationships between entities"""
        relationships = []
        
        # Common relationship patterns
        patterns = [
            (r'(\w+)\s+(?:is|are)\s+(?:a|an|the)\s+(\w+)', 'is_a'),
            (r'(\w+)\s+uses\s+(\w+)', 'uses'),
            (r'(\w+)\s+contains\s+(\w+)', 'contains'),
            (r'(\w+)\s+related to\s+(\w+)', 'related_to'),
            (r'(\w+)\s+improves\s+(\w+)', 'improves'),
            (r'(\w+)\s+processes\s+(\w+)', 'processes'),
        ]
        
        for pattern, relation in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity1, entity2 = match.groups()
                if entity1 in entities or entity2 in entities:
                    relationships.append((entity1, relation, entity2))
        
        return relationships
    
    def add_relationship(self, source_entity: str, relation: str, target_entity: str, doc_source: str = ""):
        """Manually add a relationship"""
        self.entities.add(source_entity)
        self.entities.add(target_entity)
        self.relationships[source_entity].append((relation, target_entity, doc_source))
    
    def get_related_entities(self, entity: str, relation_filter: str = None) -> List[Tuple[str, str, str]]:
        """Get all entities related to a given entity"""
        if entity not in self.relationships:
            return []
        
        related = self.relationships[entity]
        if relation_filter:
            related = [r for r in related if r[1] == relation_filter]
        
        return related
    
    def get_entity_documents(self, entity: str) -> Set[int]:
        """Get document indices mentioning an entity"""
        return self.entity_documents.get(entity, set())
    
    def search_by_entity(self, entity_keyword: str) -> List[str]:
        """Search for entities matching a keyword"""
        keyword = entity_keyword.lower()
        matching = [e for e in self.entities if keyword in e.lower()]
        return matching
    
    def get_entity_context(self, entity: str, depth: int = 1) -> Dict[str, Any]:
        """Get entity and its connections up to specified depth"""
        context = {
            "entity": entity,
            "direct_relations": self.get_related_entities(entity),
            "documents": list(self.get_entity_documents(entity)),
            "connected_entities": []
        }
        
        # Get connected entities
        if depth > 1:
            for relation, target, _ in context["direct_relations"]:
                context["connected_entities"].append({
                    "entity": target,
                    "relation": relation
                })
        
        return context
    
    def save(self):
        """Persist knowledge graph to JSON"""
        try:
            data = {
                "entities": list(self.entities),
                "relationships": {k: [(r, t, d) for r, t, d in v] for k, v in self.relationships.items()},
                "entity_documents": {k: list(v) for k, v in self.entity_documents.items()}
            }
            
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            print(f"[INFO] Knowledge graph saved to {self.persist_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save knowledge graph: {e}")
    
    def load(self):
        """Load knowledge graph from JSON"""
        if not os.path.exists(self.persist_path):
            return
        
        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.entities = set(data.get("entities", []))
            self.relationships = defaultdict(list)
            for k, v in data.get("relationships", {}).items():
                self.relationships[k] = [(r, t, d) for r, t, d in v]
            
            self.entity_documents = defaultdict(set)
            for k, v in data.get("entity_documents", {}).items():
                self.entity_documents[k] = set(v)
            
            print(f"[INFO] Knowledge graph loaded: {len(self.entities)} entities")
        except Exception as e:
            print(f"[WARN] Failed to load knowledge graph: {e}")
