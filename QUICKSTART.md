# RAG Pipeline - Quick Reference

## Installation & Setup

```bash
# Clone repo
git clone https://github.com/yourusername/RAG_Pipeline.git
cd RAG_Pipeline

# Setup environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your GROQ_API_KEY from https://console.groq.com

# Add your documents
mkdir -p data
cp your_documents.pdf data/
```

## Running the Application

```bash
python src/app.py
```

## Interactive Commands

| Command | Effect |
|---------|--------|
| `exit` | Quit the application |
| `explain` | Show detailed scoring breakdown for next query |
| Any other text | Submit query to RAG pipeline |

## Example Queries

```
Enter your query: What is machine learning?
Enter your query: Explain attention mechanisms
Enter your query: How does BERT work?
```

## Cache & Performance

- **First run**: 60-120 seconds (building indexes)
- **Subsequent runs**: < 5 seconds (loading cached indexes)
- **Cached query response**: < 100ms (if 75%+ similar to previous)
- **New LLM call**: 5-15 seconds

## Understanding the Output

```
[INFO] Performing hybrid search...
[INFO] Found 3 relevant documents (Hybrid Search)
  [1] Score: 0.812, Type: semantic
  [2] Score: 0.754, Type: keyword
  [3] Score: 0.698, Type: kg
[INFO] Hitting LLM with hybrid search results...

Summary:
<LLM response>
```

### Scoring Breakdown

- **Weights**: Semantic (50%), Keyword (30%), Knowledge Graph (20%)
- **Combined Score**: Normalized across all retrieval methods
- **Type**: Which retrieval method found the result

## Customization

Edit `src/app.py` to adjust:

```python
# Cache similarity threshold (0.0-1.0)
set_similarity_threshold(0.75)

# Search weights
hybrid_search.set_weights(semantic=0.5, keyword=0.3, kg=0.2)

# LLM Model
llm_model="llama-3.3-70b-versatile"

# Embedding Model
embedding_model="all-MiniLM-L6-v2"

# Top K results
top_k=3
```

## Troubleshooting

### "GROQ_API_KEY not set"
- Check `.env` file exists
- Verify API key is correct
- Restart application

### "No documents loaded"
- Check `data/` directory has PDF files
- Ensure PDFs are not corrupted
- Check file permissions

### Slow performance
- First run requires index building (normal)
- Check available RAM (8GB+ recommended)
- Consider using GPU for faster embeddings

### Cache not hitting
- Similarity threshold too high (adjust in app.py)
- Query phrasing too different from cached queries
- Clear cache: Delete `faiss_store/query_cache.json`

## File Structure

```
RAG_Pipeline/
├── src/              # Main application code
├── data/             # Your PDF documents
├── faiss_store/      # Generated indexes & cache
├── .env              # API keys (keep secret!)
├── .env.example      # Template for .env
├── requirements.txt  # Dependencies
└── README.md         # Full documentation
```

## Need Help?

- See full [README.md](README.md) for detailed docs
- Check [CONTRIBUTING.md](CONTRIBUTING.md) to contribute
- Review code comments in `src/` for implementation details

## Performance Tips

1. **Reduce document size** if experiencing slowness
2. **Use GPU** for faster embeddings (CUDA support)
3. **Adjust cache threshold** for better hit rate
4. **Use "explain" mode** to debug search quality
5. **Monitor console logs** for "[INFO]" and "[WARN]" messages

---

**Happy searching! 🔍**
