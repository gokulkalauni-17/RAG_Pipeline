from graphviz import Digraph

dot = Digraph("Enterprise_RAG_Pipeline", format="png")
dot.attr(rankdir="TB", bgcolor="white", fontname="Helvetica", size="16,12")
dot.attr("graph", splines="curved")

# ======================== USER INPUT LAYER ========================
with dot.subgraph(name="cluster_user") as c:
    c.attr(label="1. User Input Layer", style="rounded,bold", color="darkblue", fontcolor="darkblue")
    c.node("user", "User Query", shape="oval", style="filled", fillcolor="#E3F2FD", penwidth="2")

# ======================== CACHE LAYER ========================
with dot.subgraph(name="cluster_cache") as c:
    c.attr(label="2. Semantic Cache Layer (FAISS)", style="rounded,bold", color="darkgreen", fontcolor="darkgreen")
    c.node("cache_check", "Check Semantic Cache\n(65% similarity threshold)", shape="diamond", style="filled", fillcolor="#C8E6C9", penwidth="2")
    c.node("cache_hit", "Cache Hit ✓\nReturn Cached Result", shape="box", style="filled", fillcolor="#81C784", penwidth="2")

# ======================== QUERY PROCESSING LAYER ========================
with dot.subgraph(name="cluster_query_proc") as c:
    c.attr(label="3. Query Processing", style="rounded,bold", color="darkorange", fontcolor="darkorange")
    c.node("rewrite", "Query Rewriting\n(Semantic Clarity)", shape="box", style="filled", fillcolor="#FFE0B2", penwidth="2")
    c.node("embed", "Embed Query\n(all-MiniLM-L6-v2)", shape="box", style="filled", fillcolor="#FFE0B2", penwidth="2")

# ======================== RETRIEVAL LAYER ========================
with dot.subgraph(name="cluster_retrieval") as c:
    c.attr(label="4. Hybrid Retrieval (50% Semantic + 30% Keyword + 20% KG)", style="rounded,bold", color="purple", fontcolor="purple")
    c.node("semantic", "Semantic Search\n(FAISS Vector DB)", shape="box", style="filled", fillcolor="#E1BEE7", penwidth="2")
    c.node("keyword", "Keyword Search\n(TF-IDF)", shape="box", style="filled", fillcolor="#E1BEE7", penwidth="2")
    c.node("kg_search", "Knowledge Graph\n(Entity Relations)", shape="box", style="filled", fillcolor="#E1BEE7", penwidth="2")
    c.node("hybrid_merge", "Hybrid Merge\n(Weighted Scoring)", shape="box", style="filled", fillcolor="#CE93D8", penwidth="2")

# ======================== VALIDATION LAYER ========================
with dot.subgraph(name="cluster_validation") as c:
    c.attr(label="5. Document Validation & Grading", style="rounded,bold", color="darkred", fontcolor="darkred")
    c.node("grade", "Grade Documents\nfor Relevance\n(LLM Classifier)", shape="box", style="filled", fillcolor="#FFCCBC", penwidth="2")
    c.node("grade_decision", "Relevant?", shape="diamond", style="filled", fillcolor="#FF7043", penwidth="2")

# ======================== KNOWLEDGE STORAGE LAYER ========================
with dot.subgraph(name="cluster_knowledge") as c:
    c.attr(label="6. Enterprise Knowledge Storage", style="rounded,bold", color="navy", fontcolor="navy")
    c.node("doc_data", "Document Data\n(PDFs, Docs)", shape="folder", style="filled", fillcolor="#BBDEFB", penwidth="2")
    c.node("faiss_vec", "FAISS Vector Index\n(Persistent)", shape="cylinder", style="filled", fillcolor="#90CAF9", penwidth="2")
    c.node("kg", "Knowledge Graph\n(Entity Relations)", shape="cylinder", style="filled", fillcolor="#90CAF9", penwidth="2")
    c.node("metadata", "Semantic Cache\n(JSON + FAISS)", shape="cylinder", style="filled", fillcolor="#90CAF9", penwidth="2")

# ======================== LLM & FALLBACK LAYER ========================
with dot.subgraph(name="cluster_llm") as c:
    c.attr(label="7. LLM & Fallback Generation", style="rounded,bold", color="green", fontcolor="green")
    c.node("llm_gen", "LLM Generation\n(Groq: llama-3.3-70b)", shape="box", style="filled", fillcolor="#A5D6A7", penwidth="2")
    c.node("fallback", "Fallback Answer\n(No Relevant Docs)", shape="box", style="filled", fillcolor="#FF9800", penwidth="2")

# ======================== OUTPUT LAYER ========================
with dot.subgraph(name="cluster_output") as c:
    c.attr(label="8. Output & Response", style="rounded,bold", color="darkviolet", fontcolor="darkviolet")
    c.node("output", "AI Generated Answer\n(Code/Explanation)", shape="oval", style="filled", fillcolor="#F3E5F5", penwidth="2")
    c.node("cache_store", "Store in Semantic Cache\n(for future queries)", shape="box", style="filled", fillcolor="#E1BEE7", penwidth="2")

# ======================== MAIN WORKFLOW EDGES ========================
# User Query Entry
dot.edge("user", "cache_check", label="New Query", color="darkblue", penwidth="2")

# Cache Path
dot.edge("cache_check", "cache_hit", label="HIT (Sim>65%)", color="darkgreen", penwidth="2")
dot.edge("cache_hit", "output", label="Return Cached", color="darkgreen", penwidth="2")

# Miss Path - Query Processing
dot.edge("cache_check", "rewrite", label="MISS", color="red", penwidth="2")
dot.edge("rewrite", "embed", color="darkorange", penwidth="1.5")

# Retrieval
dot.edge("embed", "semantic", color="purple", penwidth="1.5")
dot.edge("embed", "keyword", color="purple", penwidth="1.5")
dot.edge("embed", "kg_search", color="purple", penwidth="1.5")

# Data Access for Retrieval
dot.edge("doc_data", "faiss_vec", style="dashed", color="navy")
dot.edge("doc_data", "kg", style="dashed", color="navy")
dot.edge("faiss_vec", "semantic", style="dashed", color="purple")
dot.edge("kg", "kg_search", style="dashed", color="purple")

# Hybrid Merge
dot.edge("semantic", "hybrid_merge", color="purple", penwidth="1.5")
dot.edge("keyword", "hybrid_merge", color="purple", penwidth="1.5")
dot.edge("kg_search", "hybrid_merge", color="purple", penwidth="1.5")

# Validation
dot.edge("hybrid_merge", "grade", color="darkred", penwidth="1.5")
dot.edge("grade", "grade_decision", color="darkred", penwidth="1.5")

# Grading Decision Paths
dot.edge("grade_decision", "llm_gen", label="YES (Relevant)", color="green", penwidth="2")
dot.edge("grade_decision", "fallback", label="NO (Not Relevant)", color="orange", penwidth="2", style="dashed")

# LLM Paths
dot.edge("llm_gen", "output", color="green", penwidth="1.5")
dot.edge("fallback", "output", color="orange", penwidth="1.5")

# Cache Storage
dot.edge("output", "cache_store", color="darkgreen", penwidth="1.5", style="dashed")
dot.edge("cache_store", "metadata", style="dashed", color="navy")

# ======================== LEGEND ========================
with dot.subgraph(name="cluster_legend") as c:
    c.attr(label="Legend", style="rounded", color="gray")
    c.node("leg_flow", "→ Data Flow", shape="plaintext", fontsize="10")
    c.node("leg_store", "⇢ Storage Access", shape="plaintext", fontsize="10")

dot.render("enterprise_rag_diagram")