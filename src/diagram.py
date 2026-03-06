from graphviz import Digraph

dot = Digraph("Enterprise_RAG_Pipeline", format="png")
dot.attr(rankdir="LR", bgcolor="white", fontname="Helvetica", size="20,14")

# ======================== USER INPUT LAYER (1) ========================
with dot.subgraph(name="cluster_user") as c:
    c.attr(label="1. User Input", style="rounded,bold", color="darkblue", fontcolor="darkblue")
    c.node("user", "User Query", shape="oval", style="filled", fillcolor="#E3F2FD", penwidth="2")

# ======================== CACHE LAYER (2) ========================
with dot.subgraph(name="cluster_cache") as c:
    c.attr(label="2. Semantic Cache (FAISS)", style="rounded,bold", color="darkgreen", fontcolor="darkgreen")
    c.node("cache_check", "Check Cache\n(65% threshold)", shape="diamond", style="filled", fillcolor="#C8E6C9", penwidth="2")
    c.node("cache_hit", "Return Cached\nResult", shape="box", style="filled", fillcolor="#81C784", penwidth="2")

# ======================== QUERY PROCESSING LAYER (3) ========================
with dot.subgraph(name="cluster_query_proc") as c:
    c.attr(label="3. Query Processing", style="rounded,bold", color="darkorange", fontcolor="darkorange")
    c.node("rewrite", "Query Rewriting", shape="box", style="filled", fillcolor="#FFE0B2", penwidth="2")
    c.node("embed", "Embed Query\n(all-MiniLM-L6-v2)", shape="box", style="filled", fillcolor="#FFE0B2", penwidth="2")

# ======================== RETRIEVAL LAYER (4) ========================
with dot.subgraph(name="cluster_retrieval") as c:
    c.attr(label="4. Hybrid Retrieval (50/30/20)", style="rounded,bold", color="purple", fontcolor="purple")
    c.node("hybrid_retrieval", "Hybrid Search\n• Semantic (50%)\n• Keyword (30%)\n• Knowledge Graph (20%)", 
           shape="box", style="filled", fillcolor="#CE93D8", penwidth="2")

# ======================== VALIDATION LAYER (5) ========================
with dot.subgraph(name="cluster_validation") as c:
    c.attr(label="5. Validation", style="rounded,bold", color="darkred", fontcolor="darkred")
    c.node("grade", "Grade Relevance\n(LLM Classifier)", shape="box", style="filled", fillcolor="#FFCCBC", penwidth="2")
    c.node("grade_decision", "Relevant?", shape="diamond", style="filled", fillcolor="#FF7043", penwidth="2")

# ======================== LLM LAYER (6) ========================
with dot.subgraph(name="cluster_llm") as c:
    c.attr(label="6. LLM Generation", style="rounded,bold", color="green", fontcolor="green")
    c.node("llm_gen", "LLM Response\n(Groq: llama-3.3-70b)", shape="box", style="filled", fillcolor="#A5D6A7", penwidth="2")
    c.node("fallback", "Fallback Answer", shape="box", style="filled", fillcolor="#FF9800", penwidth="2")

# ======================== OUTPUT LAYER (7) ========================
with dot.subgraph(name="cluster_output") as c:
    c.attr(label="7. Output", style="rounded,bold", color="darkviolet", fontcolor="darkviolet")
    c.node("output", "Final Response", shape="oval", style="filled", fillcolor="#F3E5F5", penwidth="2")

# ======================== KNOWLEDGE STORAGE (Background) ========================
with dot.subgraph(name="cluster_knowledge") as c:
    c.attr(label="Knowledge Storage (Background)", style="rounded,bold", color="navy", fontcolor="navy")
    c.node("doc_data", "Documents", shape="folder", style="filled", fillcolor="#BBDEFB", penwidth="1")
    c.node("faiss_vec", "FAISS Index", shape="cylinder", style="filled", fillcolor="#90CAF9", penwidth="1")
    c.node("kg", "Knowledge Graph", shape="cylinder", style="filled", fillcolor="#90CAF9", penwidth="1")
    c.node("sem_cache", "Semantic Cache", shape="cylinder", style="filled", fillcolor="#90CAF9", penwidth="1")

# ======================== MAIN WORKFLOW ========================
# Layer 1 → 2
dot.edge("user", "cache_check", color="darkblue", penwidth="2")

# Layer 2 - Cache Hit Path
dot.edge("cache_check", "cache_hit", label="HIT", color="darkgreen", penwidth="2")
dot.edge("cache_hit", "output", color="darkgreen", penwidth="2")

# Layer 2 → 3 (Cache Miss)
dot.edge("cache_check", "rewrite", label="MISS", color="red", penwidth="2")

# Layer 3 - Query Processing
dot.edge("rewrite", "embed", color="darkorange", penwidth="1.5")

# Layer 3 → 4
dot.edge("embed", "hybrid_retrieval", color="purple", penwidth="2")

# Layer 4 → 5
dot.edge("hybrid_retrieval", "grade", color="darkred", penwidth="2")
dot.edge("grade", "grade_decision", color="darkred", penwidth="2")

# Layer 5 → 6 (Decision)
dot.edge("grade_decision", "llm_gen", label="YES", color="green", penwidth="2")
dot.edge("grade_decision", "fallback", label="NO", color="orange", penwidth="2")

# Layer 6 → 7
dot.edge("llm_gen", "output", color="green", penwidth="2")
dot.edge("fallback", "output", color="orange", penwidth="2")

# ======================== KNOWLEDGE STORAGE CONNECTIONS (Dashed) ========================
dot.edge("doc_data", "faiss_vec", style="dashed", color="gray", penwidth="1")
dot.edge("doc_data", "kg", style="dashed", color="gray", penwidth="1")
dot.edge("faiss_vec", "hybrid_retrieval", style="dashed", color="gray", penwidth="1")
dot.edge("kg", "hybrid_retrieval", style="dashed", color="gray", penwidth="1")
dot.edge("output", "sem_cache", style="dashed", color="gray", penwidth="1")

dot.render("enterprise_rag_diagram")