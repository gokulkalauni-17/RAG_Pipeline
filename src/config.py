"""
Configuration for multithreading optimization in RAG pipeline.

Adjust these values based on your hardware:
- More CPU cores? Increase max_workers
- Limited RAM? Decrease batch_size
- SSDs? Can use higher batch_size
- HDD? Reduce file_loading_workers
"""

# File loading configuration
FILE_LOADING_MAX_WORKERS = 4  # Number of parallel threads for file loading (default: 4)
# Recommended: 4-8 for optimal performance on modern CPUs

# Embedding configuration
EMBEDDING_BATCH_SIZE = 32  # Number of texts to embed per batch (default: 32)
EMBEDDING_MAX_WORKERS = 2   # Number of parallel threads for embedding batches (default: 2)
# Recommended: Keep embedding_workers <= 2 to avoid memory issues with large models

# Device for embedding model ('cpu', 'cuda', etc.).
# If set to None the pipeline will use DEVICE environment or detect automatically.
DEVICE: str = "cpu"

# Tuning tips:
# 1. FILE_LOADING_MAX_WORKERS: 
#    - Increase if you have many CPU cores and I/O is fast
#    - Decrease if you have memory constraints
#
# 2. EMBEDDING_BATCH_SIZE:
#    - Increase (64, 128) for more memory, faster processing
#    - Decrease (16, 8) for limited memory
#
# 3. EMBEDDING_MAX_WORKERS:
#    - Usually 2 is optimal (more workers = more memory usage)
#    - Increase to 3-4 only if you have abundant RAM (32GB+)
