from pathlib import Path
from typing import List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_core.documents import Document
from src.config import FILE_LOADING_MAX_WORKERS


def compute_sha256(path: Path) -> str:
    """Return SHA256 hex digest of file contents."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def get_cache_file_path(cache_dir: Path, file_path: Path) -> Path:
    name_hash = hashlib.sha256(str(file_path).encode()).hexdigest()
    return cache_dir / f"{name_hash}.pkl"


def load_single_file(file_info: Tuple[str, Path, str], cache_dir: Path) -> Tuple[List[Any], str, str]:
    """
    Load a single file with caching support.
    
    Args:
        file_info: (file_type, file_path, previous_hash)
        cache_dir: directory where cached pickles and hashes are stored
    Returns:
        (documents, file_path_str, new_hash)
    """
    file_type, file_path, prev_hash = file_info
    new_hash = compute_sha256(file_path)
    cache_file = get_cache_file_path(cache_dir, file_path)

    # if hash matches and cache exists, load from pickle
    if prev_hash and prev_hash == new_hash and cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                docs = pickle.load(f)
            print(f"[INFO] Loaded {len(docs)} docs from cache for {file_path.name}")
            return docs, str(file_path), new_hash
        except Exception as e:
            print(f"[WARN] Failed to load cache for {file_path}: {e}")
            # fall through to normal loading

    try:
        if file_type == 'pdf':
            loader = PyPDFLoader(str(file_path))
        elif file_type == 'txt':
            loader = TextLoader(str(file_path))
        elif file_type == 'csv':
            loader = CSVLoader(str(file_path))
        elif file_type == 'xlsx':
            loader = UnstructuredExcelLoader(str(file_path))
        elif file_type == 'docx':
            loader = Docx2txtLoader(str(file_path))
        elif file_type == 'json':
            # Handle JSON files manually
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    # Convert JSON to documents
                    if isinstance(json_data, list):
                        loaded = [Document(page_content=json.dumps(item), metadata={"source": str(file_path)}) for item in json_data]
                    elif isinstance(json_data, dict):
                        loaded = [Document(page_content=json.dumps(json_data), metadata={"source": str(file_path)})]
                    else:
                        loaded = [Document(page_content=str(json_data), metadata={"source": str(file_path)})]
                    print(f"[INFO] Loaded {len(loaded)} documents from {file_path.name}")
                    # cache result
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(loaded, f)
                    except Exception as e:
                        print(f"[WARN] Unable to cache {file_path}: {e}")
                    return loaded, str(file_path), new_hash
            except Exception as e:
                print(f"[ERROR] Failed to load JSON {file_path}: {e}")
                return [], str(file_path), new_hash
        else:
            return [], str(file_path), new_hash
        
        loaded = loader.load()
        print(f"[INFO] Loaded {len(loaded)} documents from {file_path.name}")
        # cache result
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(loaded, f)
        except Exception as e:
            print(f"[WARN] Unable to cache {file_path}: {e}")
        return loaded, str(file_path), new_hash
    except Exception as e:
        print(f"[ERROR] Failed to load {file_path}: {e}")
        return [], str(file_path), new_hash

def load_all_documents(data_dir: str, max_workers: int = None) -> List[Any]:
    """
    Load all supported files from the data directory in parallel, using a simple
    cache to avoid reloading unchanged files. Supported formats: PDF, TXT, CSV,
    Excel, Word, JSON.
    
    Args:
        data_dir: Path to data directory
        max_workers: Number of parallel workers (default: FILE_LOADING_MAX_WORKERS from config)
    
    Returns:
        List of loaded documents
    """
    if max_workers is None:
        max_workers = FILE_LOADING_MAX_WORKERS
    
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[INFO] Data path: {data_path}")
    documents = []
    files_to_load = []

    # prepare cache directory and load stored hashes
    cache_dir = data_path / ".cache"
    cache_dir.mkdir(exist_ok=True)
    hashes_path = cache_dir / "hashes.json"
    try:
        if hashes_path.exists():
            with open(hashes_path, 'r', encoding='utf-8') as f:
                previous_hashes = json.load(f)
        else:
            previous_hashes = {}
    except Exception:
        previous_hashes = {}

    # we'll update this map as we process files
    updated_hashes = {}

    # Collect all files to load
    pdf_files = list(data_path.glob('**/*.pdf'))
    txt_files = list(data_path.glob('**/*.txt'))
    csv_files = list(data_path.glob('**/*.csv'))
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    docx_files = list(data_path.glob('**/*.docx'))
    json_files = list(data_path.glob('**/*.json'))
    
    # Create file info tuples
    for f in pdf_files:
        prev = previous_hashes.get(str(f), '')
        files_to_load.append(('pdf', f, prev))
    for f in txt_files:
        prev = previous_hashes.get(str(f), '')
        files_to_load.append(('txt', f, prev))
    for f in csv_files:
        prev = previous_hashes.get(str(f), '')
        files_to_load.append(('csv', f, prev))
    for f in xlsx_files:
        prev = previous_hashes.get(str(f), '')
        files_to_load.append(('xlsx', f, prev))
    for f in docx_files:
        prev = previous_hashes.get(str(f), '')
        files_to_load.append(('docx', f, prev))
    for f in json_files:
        prev = previous_hashes.get(str(f), '')
        files_to_load.append(('json', f, prev))
    
    total_files = len(files_to_load)
    print(f"[INFO] Found {total_files} files. Loading with {max_workers} workers...")
    
    if total_files == 0:
        print("[INFO] No documents found in data directory.")
        return documents
    
    # Load files in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(load_single_file, file_info, cache_dir): file_info 
                   for file_info in files_to_load}
        
        for future in as_completed(futures):
            try:
                loaded_docs, file_path, file_hash = future.result()
                documents.extend(loaded_docs)
                updated_hashes[file_path] = file_hash
            except Exception as e:
                print(f"[ERROR] Error loading file: {e}")

    # persist updated hashes
    try:
        with open(hashes_path, 'w', encoding='utf-8') as f:
            json.dump(updated_hashes, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] Unable to write hashes file: {e}")
    
    print(f"[INFO] Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
    
    # Example with custom number of workers
    docs_custom = load_all_documents("data", max_workers=6)
    print(f"Loaded {len(docs_custom)} documents with custom workers.")