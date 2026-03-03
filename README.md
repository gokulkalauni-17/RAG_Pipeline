# RAG Pipeline with Semantic Caching & Advanced Retrieval

A production-ready **Retrieval-Augmented Generation (RAG)** system combining semantic search, intelligent caching, and advanced document retrieval with **FAISS**, **Groq LLM**, and **LangGraph**.

## 🎯 Key Features

### Core RAG Capabilities
- **Hybrid Search**: Combines semantic (50%), keyword (30%), and knowledge graph (20%) retrieval
- **FAISS Vector Store**: Fast similarity search with persistent indexing
- **Groq LLM Integration**: Using `llama-3.3-70b-versatile` for summarization
- **Knowledge Graph**: Entity-relationship extraction from documents

### Advanced Intelligent Features
- **Semantic Caching with FAISS**: Sub-millisecond cache hits for similar queries (configurable threshold)
- **Document Relevance Grading**: Binary classification to validate retrieved documents
- **Automatic Query Rewriting**: Improves semantic clarity when documents aren't relevant
- **Retry Logic**: Up to 2 automatic attempts with rewritten queries
- **Fallback Answer Generation**: Graceful handling when knowledge base lacks information
- **Persistent Cache**: Survives application restarts with JSON metadata + FAISS index

---

## 📦 Installation

### Prerequisites

- Python 3.8+
- 8GB+ RAM recommended
- GPU optional (CUDA support for faster embeddings)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/RAG_Pipeline.git
cd RAG_Pipeline
```

2. **Create a Python virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
Create a `.env` file in the project root using `.env.example` as a template:
```bash
cp .env.example .env
# Edit .env and add your Groq API key
```

Get your Groq API key from: https://console.groq.com

⚠️ **Security Note:** Never commit `.env` to Git. It's already in `.gitignore`.

5. **Prepare document data**
```bash
# Create data directory and add PDF files
mkdir -p data
# Copy your PDF files to the data/ folder
```

6. **First run will build indexes**
- This may take 1-2 minutes depending on document size
- Subsequent runs load cached indexes instantly

---

## 🚀 Quick Start

### Run the Application

```bash
python src/app.py
```

### First Run
- **Startup time:** 60-120 seconds (building indexes)
- **Status:** Watch console for "[INFO]" messages

### Subsequent Runs
- **Startup time:** 5-10 seconds (loading cached indexes)
- **Status:** Displays "loading existing index" messages

### Example Usage

```
[INFO] Loading documents...
[INFO] Loaded 150 documents.
[INFO] Building vector store...
[INFO] Vector store hash matches current documents; loading existing index...
[INFO] Building keyword search index...
[INFO] Building knowledge graph...
[INFO] Initializing hybrid search...

Enter your query (type 'exit' to quit, 'explain' for detailed scoring): 
What is machine learning?

[INFO] Performing hybrid search...
[INFO] Found 3 relevant documents (Hybrid Search)
  [1] Score: 0.891, Type: hybrid
  [2] Score: 0.754, Type: hybrid
  [3] Score: 0.621, Type: hybrid
[INFO] Hitting LLM with hybrid search results...

Summary:
Machine learning is a subset of artificial intelligence that enables systems 
to learn and improve from experience without being explicitly programmed...

Enter your query (type 'exit' to quit, 'explain' for detailed scoring): exit
Exiting the application.
```

### Special Commands

| Command | Action |
|---------|--------|
| `exit` | Quit the application |
| `explain` | Show detailed scoring breakdown for next query |

---

## 📖 Architecture

### System Components

```
┌─────────────────────────────────────────────────────┐
│                    app.py                           │
│           Main application orchestrator            │
│  • Startup & index management                      │
│  • Query loop & user interaction                   │
│  • Result caching coordination                     │
└─────────────────────────────────────────────────────┘
         ↓              ↓              ↓
    ┌─────────┐   ┌─────────────┐  ┌──────────────┐
    │Vector   │   │  Keyword    │  │ Knowledge    │
    │Store    │   │  Search     │  │ Graph        │
    │(FAISS)  │   │(Inverted)   │  │(Entities)    │
    └─────────┘   └─────────────┘  └──────────────┘
         ↓              ↓              ↓
    └─────────────────────────────────────────┘
           Hybrid Search (Combined)
    └─────────────────────────────────────────┘
                      ↓
              ┌─────────────────┐
              │  Cache Check    │
              │  (Similarity)   │
              └────────┬────────┘
                 ↙      ↓      ↘
            [Cache Hit] [Query LLM] [Cache Miss]
                 ↓      ↓           ↓
                 └──→ Groq LLM ←──┘
                      ↓
              ┌──────────────────┐
              │  Cache Result    │
              │  Display Answer  │
              └──────────────────┘
```

### Module Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **app.py** | Orchestration & CLI | Main application loop, startup logic |
| **data_loader.py** | Document ingestion | Load PDFs with caching, parallel processing |
| **embedding.py** | Text vectorization | Create embeddings using sentence-transformers |
| **vectorstore.py** | Vector index management | Build/load FAISS indexes, similarity search |
| **keyword_search.py** | Term-based retrieval | Inverted index, TF-IDF scoring |
| **knowledge_graph.py** | Entity relationships | Extract entities, build graph, search by relationships |
| **hybrid_search.py** | Result combination | Combine 3 methods with weights, normalize scores |
| **rag_search.py** | LLM integration | Generate answers using Groq API |
| **cache.py** | Query result caching | Semantic similarity matching, persistent storage |
| **sanitizer.py** | Input validation | Remove injection vectors, normalize text |
| **config.py** | Configuration | Multithreading & device settings |

---

## ⚙️ Configuration

### Hybrid Search Weights

Adjust in `src/app.py` (line ~115):

```python
hybrid_search.set_weights(semantic=0.5, keyword=0.3, kg=0.2)
```

**Recommendations:**
- **Technical documents:** `semantic=0.6, keyword=0.2, kg=0.2`
- **Named entity queries:** `semantic=0.3, keyword=0.3, kg=0.4`
- **General queries:** `semantic=0.5, keyword=0.3, kg=0.2` (default)

### Cache Similarity Threshold

In `src/app.py` (line ~50):

```python
set_similarity_threshold(0.90)  # 0.0 to 1.0
```

- **Higher (0.95+):** Stricter matching, fewer cache hits, more fresh LLM calls
- **Lower (0.80):** Relaxed matching, more cache hits, potentially stale answers

### Embedding Model

In `src/app.py` (line ~60):

```python
FaissVectorStore(embedding_model="all-MiniLM-L6-v2")
```

**Available Models:**
- `all-MiniLM-L6-v2` (default) - Fast, lightweight, 384 dims
- `all-mpnet-base-v2` - Slower, better quality, 768 dims

### Performance Tuning

Edit `src/config.py`:

```python
FILE_LOADING_MAX_WORKERS = 4          # Parallel file readers
EMBEDDING_BATCH_SIZE = 32             # Texts per embedding batch
EMBEDDING_MAX_WORKERS = 2             # Parallel embedding workers
DEVICE = "cpu"                        # "cpu" or "cuda"
```

**Recommendations:**
- **Limited RAM (<8GB):** Reduce `EMBEDDING_BATCH_SIZE` to 16
- **GPU Available:** Set `DEVICE = "cuda"` for 3-5x speedup
- **Many cores (16+):** Increase `FILE_LOADING_MAX_WORKERS` to 8

---

## 🔄 Complete Workflow

### Phase 1: Startup (One-time or on document change)

```
1. Load Documents
   └─ Read PDFs from data/ folder with file-level caching

2. Compute Document Hash
   └─ SHA256 of all content for change detection

3. Load Previous Build State
   └─ Check faiss_store/build_state.json

4. Smart Vector Store
   ├─ If hash matches: Load existing FAISS index ⚡
   └─ Else: Build embeddings & index 🔄

5. Build Keyword Index
   └─ Create inverted term mapping

6. Smart Knowledge Graph
   ├─ If hash matches: Load existing graph ⚡
   └─ Else: Extract entities & relationships 🔄

7. Initialize Hybrid Search
   └─ Configure weights & combine methods
```

### Phase 2: Query Processing (For each user query)

```
1. Check Cache
   ├─ If similarity ≥ 90%: Return cached answer ⚡⚡
   └─ Else: Continue to search

2. Hybrid Search
   ├─ Semantic: Vector similarity (FAISS)
   ├─ Keyword: Term matching (TF-IDF)
   └─ Knowledge Graph: Entity traversal
   
3. Combine Results
   └─ Weighted scoring: sem(50%) + kw(30%) + kg(20%)

4. Extract Context
   └─ Get top-3 documents, sanitize, limit to 3000 chars

5. Call LLM
   └─ Groq API (llama-3.3-70b-versatile)

6. Cache Result
   └─ Store for future similar queries

7. Display Answer
   └─ Show summary to user
```

---

## 📊 Performance Metrics

### Timing Benchmarks

| Scenario | Time | Notes |
|----------|------|-------|
| **First Run** | 60-120s | Building all 3 indexes |
| **Cached Startup** | 5-10s | Loading existing indexes |
| **Cache Hit** | <100ms | Return stored answer |
| **Hybrid Search** | ~2s | 3 retrieval methods |
| **LLM Generation** | ~3-5s | Groq API latency |
| **Total (Cache Miss)** | ~5-10s | Search + LLM |

### Memory Usage

| Component | Memory | Notes |
|-----------|--------|-------|
| **FAISS Index** | ~100MB | 10,000 documents |
| **Knowledge Graph** | ~10MB | JSON file |
| **Embeddings (cached)** | ~150MB | In-memory for queries |
| **Total** | ~300MB | Typical usage |

### Scalability

- **Documents:** 1,000-100,000+
- **Query latency:** 5-10s (consistent, cached)
- **Index rebuild:** Linear with document count

---

## 🔧 Troubleshooting

### "GROQ_API_KEY not set"

**Solution:**
```bash
# Create .env file in project root
echo "GROQ_API_KEY=your_key_here" > .env
```

Get key from: https://console.groq.com

### "No documents found in data directory"

**Solution:**
```bash
# Create data folder
mkdir -p data

# Add PDF files
cp your_documents/*.pdf data/
```

Application exits if no documents found.

### "Vector store hash matches but index missing"

**Solution:**
```bash
# Delete corrupted state file to force rebuild
rm -f faiss_store/build_state.json
python src/app.py  # Rebuild all indexes
```

### Slow Performance

**Check:**
1. Are you using CPU only? Enable CUDA:
```python
# In config.py
DEVICE = "cuda"
```

2. Is RAM limited? Reduce batch sizes:
```python
EMBEDDING_BATCH_SIZE = 16  # from 32
EMBEDDING_MAX_WORKERS = 1  # from 2
```

3. First run is always slow (60-120s) - this is normal

### Inaccurate Results

**Adjust weights:**
```python
# For more entity-based queries
hybrid_search.set_weights(semantic=0.4, keyword=0.3, kg=0.3)

# For technical queries
hybrid_search.set_weights(semantic=0.6, keyword=0.2, kg=0.2)
```

---

## 📁 Project Structure

```
RAG_Pipeline/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env                              # API keys (create manually)
│
├── src/
│   ├── __init__.py
│   ├── app.py                        # Main application
│   ├── config.py                     # Configuration
│   ├── data_loader.py                # Document loading
│   ├── embedding.py                  # Vectorization
│   ├── vectorstore.py                # FAISS index
│   ├── keyword_search.py             # Keyword retrieval
│   ├── knowledge_graph.py            # Entity graph
│   ├── hybrid_search.py              # Combined search
│   ├── rag_search.py                 # LLM integration
│   ├── cache.py                      # Result caching
│   └── sanitizer.py                  # Input validation
│
├── data/                             # Your documents (PDFs)
│   └── *.pdf
│
└── faiss_store/                      # Generated indexes
    ├── faiss.index                   # Vector index
    ├── metadata.json                 # Vector metadata
    ├── knowledge_graph.json          # Entity graph
    ├── query_cache.json              # Cached results
    └── build_state.json              # Build state tracking
```

---

## 🚦 Usage Examples

### Example 1: Basic Query

```bash
$ python src/app.py

[INFO] Loading documents...
[INFO] Loaded 150 documents.
...

Enter your query: What is transformer architecture?
[INFO] Performing hybrid search...
[INFO] Found 3 relevant documents

Summary:
A transformer is a neural network architecture based on self-attention mechanisms...
```

### Example 2: Detailed Scoring

```bash
Enter your query: explain
Enter your query: How do neural networks learn?

[DEBUG] Explanation:
  Query: How do neural networks learn?
  Weights - Semantic: 0.50, Keyword: 0.30, KG: 0.20
  Semantic Results: 5
  Keyword Results: 3
  KG Results: 2

Summary:
Neural networks learn through backpropagation...
```

### Example 3: Cache Hit

```bash
Enter your query: What is machine learning?
[CACHE] Summary (from cache - 95.2% match):
Machine learning is a subset of artificial intelligence...

(Returned in <100ms from cache)
```

---

## 📚 Technical Details

### Embedding Model
- **Model:** `all-MiniLM-L6-v2` by SENTENCE-TRANSFORMERS
- **Dimensions:** 384
- **Speed:** ~1000 texts/sec
- **Accuracy:** Optimized for semantic search

### Vector Search
- **Library:** FAISS (Facebook AI Similarity Search)
- **Index Type:** IndexFlatL2 (brute force with L2 distance)
- **Speed:** Milliseconds for 100K vectors
- **Quality:** Exact nearest neighbors

### Knowledge Graph
- **Format:** Adjacency list (JSON)
- **Entity Extraction:** Regex patterns + NLP
- **Search:** Breadth-first traversal (depth 1-2)
- **Size:** ~10-100MB for 10K documents

### Hybrid Scoring
- **Normalization:** Min-max scaling to [0, 1]
- **Combination:** Weighted sum
- **Formula:** final_score = w_sem × sem + w_kw × kw + w_kg × kg

### Caching
- **Backend:** In-memory dict + JSON persistence
- **Similarity:** Cosine distance on embeddings
- **Threshold:** 90% by default (configurable)
- **Storage:** faiss_store/query_cache.json

### LLM Integration
- **Provider:** Groq (fast, efficient)
- **Model:** llama-3.3-70b-versatile
- **Temperature:** 0.7 (balanced)
- **Max tokens:** 1024
- **Cost:** ~$0.0002 per query

---

## ⚡ Performance Tips

1. **First Run:** 60-120s is normal (build indexes)
2. **Subsequent Runs:** Use cached indexes (5-10s)
3. **Cache Hits:** <100ms (return stored answer)
4. **GPU:** 3-5x faster embeddings if available
5. **Batch Size:** Balance speed vs memory

---

## 🔐 Security

- **Input Sanitization:** Removes prompt injection patterns
- **API Keys:** Stored in .env (not in code)
- **Context Truncation:** Limits to 3000 chars (prevents token overflow)
- **Query Limits:** Sanitized to 500 chars

---

## 📝 Dependencies

```
langchain                 # LLM framework
langchain-core           # Core abstractions
langchain-community      # Community integrations
pypdf                    # PDF extraction
sentence-transformers    # Embeddings
faiss-cpu               # Vector search
langchain-groq          # Groq integration
python-dotenv           # Environment variables
scikit-learn            # ML utilities
psutil                  # System monitoring
neo4j                   # Graph DB (optional)
```

---

## 🤝 Contributing

To improve the pipeline:

1. Modify weights in `hybrid_search.py` for better accuracy
2. Add new embedding models in `embedding.py`
3. Extend entity extraction in `knowledge_graph.py`
4. Add custom sanitization rules in `sanitizer.py`

---

## 📄 License

This project is part of the Learning/Python collection.

---

## 🆘 Support

**Issues?**
1. Check troubleshooting section above
2. Verify documents in `data/` folder
3. Confirm `.env` has valid `GROQ_API_KEY`
4. Check console for `[ERROR]` messages
5. Try deleting `faiss_store/build_state.json` to rebuild

**Questions?**
- Review the RAG Pipeline architecture documentation
- Check inline code comments
- See example workflows above

---

## � Project Structure

```
RAG_Pipeline/
├── src/
│   ├── app.py                 # Main application entry point
│   ├── search.py              # RAG search implementation
│   ├── vectorstore.py         # FAISS vector store
│   ├── cache.py               # Query result caching
│   ├── hybrid_search.py       # Hybrid search orchestration
│   ├── knowledge_graph.py     # Knowledge graph builder
│   ├── keyword_search.py      # Keyword search implementation
│   ├── embedding.py           # Embedding pipeline
│   ├── data_loader.py         # Document loading
│   ├── sanitizer.py           # Input sanitization
│   └── config.py              # Configuration management
├── data/                       # User documents (PDF files) - Git ignored
├── faiss_store/                # FAISS index & cache - Git ignored
├── .env                        # Environment variables - Git ignored
├── .gitignore                  # Git ignore rules
├── LICENSE                     # MIT License
├── CONTRIBUTING.md             # Contribution guidelines
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### Important Notes

- **`data/` directory**: Add your PDF documents here. This directory is ignored by Git for privacy.
- **`faiss_store/`**: Contains FAISS indexes, cache, and knowledge graph. These are generated on first run and ignored by Git.
- **`.env` file**: Contains API keys. Always kept secure and ignored by Git.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:
- Code style and standards
- Setting up development environment
- Submitting pull requests
- Types of contributions we're looking for

---

## 📈 Roadmap

- [ ] Multi-language support
- [ ] Custom embedding models
- [ ] Persistent graph database backend
- [ ] Web UI (FastAPI)
- [ ] Streaming responses
- [ ] Batch query processing

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

Gokul Kalauni

---

**Star ⭐ this repository if you find it helpful!**
- [ ] Analytics dashboard

---

**Last Updated:** March 1, 2026

