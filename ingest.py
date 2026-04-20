"""
╔══════════════════════════════════════════════════════════════════════╗
║              MindVault v2 — Ingestion Pipeline (ingest.py)           ║
║  Reads local files → chunks → embeds → stores in ChromaDB            ║
║  Also extracts TF-IDF keywords per chunk for the Visual Graph tab.   ║
╚══════════════════════════════════════════════════════════════════════╝

Supported: .txt  .md  .pdf  .docx  .eml
Usage:
    python ingest.py --folder ~/Documents/Archive
    python ingest.py --folder ~/Documents/Archive --reset
"""

import os
import sys
import logging
import argparse
import re
from pathlib import Path
from datetime import datetime
from collections import Counter

# ── stdlib check ──────────────────────────────────────────────────────
try:
    import requests
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn,
        TextColumn, TaskProgressColumn,
    )
    from tqdm import tqdm
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
except ImportError as e:
    print(f"[ERROR] Missing dependency: {e}\nRun: pip install -r requirements.txt")
    sys.exit(1)

try:
    from langchain_community.document_loaders import (
        TextLoader,
        UnstructuredMarkdownLoader,
        PyPDFLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredEmailLoader,
    )
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_ollama import OllamaEmbeddings
    from langchain_chroma import Chroma
except ImportError as e:
    print(f"[ERROR] LangChain import failed: {e}\nRun: pip install -r requirements.txt")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────
# Configuration — edit these constants to change behaviour
# ─────────────────────────────────────────────────────────────────────
CHROMA_PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME    = "mindvault_archive"
OLLAMA_BASE_URL    = "http://localhost:11434"
EMBED_MODEL        = "llama3.2"

CHUNK_SIZE    = 1000   # characters per chunk
CHUNK_OVERLAP = 200    # overlap to preserve sentence context

# Number of TF-IDF keywords to extract per chunk (stored as metadata
# so the Visual Graph tab can build the concept map without re-reading)
KEYWORDS_PER_CHUNK = 8

# Stopwords for TF-IDF (minimal set — keeps keyword extraction fast
# without requiring NLTK downloads)
STOPWORDS = {
    "the","a","an","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall",
    "this","that","these","those","it","its","i","we","you","he","she",
    "they","them","their","our","my","your","his","her","its","not","no",
    "up","out","so","by","from","as","if","then","than","because","while",
    "about","into","through","during","before","after","above","below",
    "between","each","all","both","few","more","most","other","some","such",
    "own","same","just","also","when","where","which","who","how","what",
    "very","can","here","there","us","me","him","her","them","any","only",
}

LOADER_MAP = {
    ".txt":  TextLoader,
    ".md":   UnstructuredMarkdownLoader,
    ".pdf":  PyPDFLoader,
    ".docx": UnstructuredWordDocumentLoader,
    ".eml":  UnstructuredEmailLoader,
}

console = Console()

logging.basicConfig(
    level=logging.WARNING,
    handlers=[logging.FileHandler(Path(__file__).parent / "mindvault.log")],
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Pre-flight helpers
# ─────────────────────────────────────────────────────────────────────

def check_ollama() -> bool:
    try:
        return requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5).status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def check_model(model: str) -> bool:
    try:
        resp   = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
        return model.split(":")[0] in models
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────
# Step 1 — Discover files
# ─────────────────────────────────────────────────────────────────────

def discover_files(folder: Path) -> list[Path]:
    """Recursively find all supported files under `folder`."""
    files = []
    for ext in LOADER_MAP:
        files.extend(folder.rglob(f"*{ext}"))
    return sorted(files)


# ─────────────────────────────────────────────────────────────────────
# Step 2 — Load individual file
# ─────────────────────────────────────────────────────────────────────

def load_file(path: Path) -> list:
    """
    Load a single file into LangChain Document objects.
    Adds rich metadata: source, file_name, file_type, date_modified.
    The date_modified is stored as both ISO string (for display) and
    a plain YYYY-MM prefix so Echo can group writings by month.
    """
    ext = path.suffix.lower()
    cls = LOADER_MAP.get(ext)
    if cls is None:
        return []
    try:
        loader = cls(str(path), encoding="utf-8", autodetect_encoding=True) \
                 if cls == TextLoader else cls(str(path))
        docs = loader.load()
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        for doc in docs:
            doc.metadata.update({
                "source":        str(path),
                "file_name":     path.name,
                "file_type":     ext,
                "date_modified": mtime.isoformat(),
                "year_month":    mtime.strftime("%Y-%m"),  # for Echo timeline grouping
            })
        return docs
    except Exception as exc:
        logger.warning(f"Load failed: {path}: {exc}")
        return []


# ─────────────────────────────────────────────────────────────────────
# Step 3 — Chunk
# ─────────────────────────────────────────────────────────────────────

def chunk_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks.
    Separator hierarchy: paragraph → newline → sentence → word → char.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


# ─────────────────────────────────────────────────────────────────────
# Keyword Extraction (TF-IDF over all chunks)
# ─────────────────────────────────────────────────────────────────────

def extract_keywords_tfidf(chunks: list) -> list[list[str]]:
    """
    Run TF-IDF across all chunks and return the top KEYWORDS_PER_CHUNK
    keywords for each chunk.

    Why TF-IDF here instead of an LLM call?
      - Zero latency: computed in-process.
      - No Ollama request needed during ingestion.
      - Keywords are stored once in ChromaDB metadata so the Graph tab
        can read them instantly without re-embedding.

    Returns a list of keyword lists, one per chunk (same order).
    """
    if not chunks:
        return []

    texts = [c.page_content for c in chunks]

    # Custom tokeniser: lowercase alpha words of length ≥3
    def tokenise(text: str) -> str:
        words = re.findall(r"[a-z]{3,}", text.lower())
        return " ".join(w for w in words if w not in STOPWORDS)

    cleaned = [tokenise(t) for t in texts]

    # Fit TF-IDF — use unigrams + bigrams for richer concepts
    vec = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=1,
    )
    try:
        tfidf_matrix = vec.fit_transform(cleaned)
    except ValueError:
        # All texts empty after cleaning
        return [[] for _ in chunks]

    feature_names = np.array(vec.get_feature_names_out())
    keywords_per_chunk = []

    for row_idx in range(tfidf_matrix.shape[0]):
        row   = tfidf_matrix[row_idx].toarray().flatten()
        # argsort descending, take top N
        top_n = int(np.argsort(row)[::-1][:KEYWORDS_PER_CHUNK])
        kws   = [feature_names[i] for i in np.argsort(row)[::-1][:KEYWORDS_PER_CHUNK]
                 if row[i] > 0]
        keywords_per_chunk.append(kws)

    return keywords_per_chunk


# ─────────────────────────────────────────────────────────────────────
# Step 4 — Embed and store in ChromaDB
# ─────────────────────────────────────────────────────────────────────

def embed_and_store(chunks: list, keywords_per_chunk: list[list[str]],
                    reset: bool = False) -> "Chroma":
    """
    Attach keyword metadata to each chunk, then embed and persist in ChromaDB.

    ChromaDB stores the embeddings so we never re-embed unless --reset.
    The `keywords` field (comma-separated string) is what the Visual Graph
    tab reads when building the concept network.
    """
    console.print("\n[bold cyan]Initialising embedding model...[/bold cyan]")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)

    # Attach keyword metadata before storing
    for chunk, kws in zip(chunks, keywords_per_chunk):
        chunk.metadata["keywords"] = ", ".join(kws)  # CSV string → easy to split later

    if reset:
        console.print("[yellow]--reset: deleting existing collection...[/yellow]")
        import chromadb as _chromadb
        client = _chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        try:
            client.delete_collection(COLLECTION_NAME)
            console.print("[green]✓ Collection cleared.[/green]")
        except Exception:
            pass

    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    BATCH = 50
    total = len(chunks)
    console.print(f"[bold cyan]Embedding {total} chunks (batch={BATCH})...[/bold cyan]\n")

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as prog:
        task = prog.add_task("Embedding & storing…", total=total)
        for i in range(0, total, BATCH):
            vs.add_documents(chunks[i : i + BATCH])
            prog.advance(task, len(chunks[i : i + BATCH]))

    return vs


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MindVault v2 — Ingest local documents into ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python ingest.py --folder ~/Documents/Archive
  python ingest.py --folder ~/Documents/Archive --reset
  python ingest.py --folder ~/Documents/Archive --model mistral
        """,
    )
    parser.add_argument("--folder", required=True,
                        help="Folder containing your documents")
    parser.add_argument("--reset", action="store_true",
                        help="Wipe existing index and start fresh")
    parser.add_argument("--model", default=EMBED_MODEL,
                        help=f"Ollama model for embeddings (default: {EMBED_MODEL})")
    args = parser.parse_args()

    # Override model if CLI flag provided
    global EMBED_MODEL
    EMBED_MODEL = args.model

    # ── Banner ────────────────────────────────────────────────────────
    console.print(Panel.fit(
        "[bold magenta]MindVault v2[/bold magenta] — Privacy-First Personal Archivist\n"
        "[dim]100% local · No cloud · Your data stays on this machine.[/dim]",
        border_style="magenta",
    ))

    # ── Pre-flight ────────────────────────────────────────────────────
    console.print("\n[bold]Pre-flight checks[/bold]")
    if not check_ollama():
        console.print("[bold red]✗ Ollama not running. Start with: ollama serve[/bold red]")
        sys.exit(1)
    console.print("[green]✓ Ollama online.[/green]")

    if not check_model(EMBED_MODEL):
        console.print(f"[red]✗ Model '{EMBED_MODEL}' not found.[/red]")
        console.print(f"  Pull with: [yellow]ollama pull {EMBED_MODEL}[/yellow]")
        sys.exit(1)
    console.print(f"[green]✓ Model '{EMBED_MODEL}' ready.[/green]")

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        console.print(f"[red]✗ Folder not found: {folder}[/red]")
        sys.exit(1)
    console.print(f"[green]✓ Archive folder:[/green] {folder}\n")

    # ── 1. Discover ───────────────────────────────────────────────────
    console.print("[bold]1 / 5 — Discovering files...[/bold]")
    files = discover_files(folder)
    if not files:
        console.print("[yellow]No supported files found.[/yellow]")
        sys.exit(0)

    tbl = Table(title=f"{len(files)} file(s) found", header_style="bold cyan")
    tbl.add_column("File");  tbl.add_column("Type", width=7); tbl.add_column("Size", width=10)
    for f in files[:20]:
        tbl.add_row(f.name, f.suffix, f"{f.stat().st_size/1024:.1f} KB")
    if len(files) > 20:
        tbl.add_row(f"… and {len(files)-20} more", "", "")
    console.print(tbl)

    # ── 2. Load ───────────────────────────────────────────────────────
    console.print("\n[bold]2 / 5 — Loading documents...[/bold]")
    all_docs, failed = [], []
    for f in tqdm(files, desc="Loading", unit="file"):
        docs = load_file(f)
        (all_docs if docs else failed).extend(docs if docs else [f.name])
    console.print(f"[green]✓ {len(all_docs)} pages loaded. "
                  f"{'⚠ ' + str(len(failed)) + ' failed.' if failed else ''}[/green]")

    # ── 3. Chunk ──────────────────────────────────────────────────────
    console.print("\n[bold]3 / 5 — Chunking...[/bold]")
    chunks = chunk_documents(all_docs)
    console.print(f"[green]✓ {len(chunks)} chunks "
                  f"({CHUNK_SIZE} chars, {CHUNK_OVERLAP} overlap).[/green]")

    # ── 4. Keyword Extraction ─────────────────────────────────────────
    console.print("\n[bold]4 / 5 — Extracting TF-IDF keywords for Concept Graph...[/bold]")
    keywords = extract_keywords_tfidf(chunks)
    # Summarise top global keywords for the user
    all_kws = [kw for chunk_kws in keywords for kw in chunk_kws]
    top10   = Counter(all_kws).most_common(10)
    console.print(f"[green]✓ Keywords extracted.[/green] "
                  f"Top global: {', '.join(k for k,_ in top10)}")

    # ── 5. Embed & Store ──────────────────────────────────────────────
    console.print("\n[bold]5 / 5 — Embedding & storing in ChromaDB...[/bold]")
    vs = embed_and_store(chunks, keywords, reset=args.reset)

    console.print(Panel(
        f"[bold green]✓ Ingestion complete![/bold green]\n\n"
        f"  Chunks in ChromaDB : [cyan]{vs._collection.count()}[/cyan]\n"
        f"  Persist dir        : [cyan]{CHROMA_PERSIST_DIR}[/cyan]\n"
        f"  Keywords / chunk   : [cyan]{KEYWORDS_PER_CHUNK}[/cyan]\n\n"
        f"  [dim]Run [yellow]streamlit run app.py[/yellow] to start MindVault.[/dim]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
