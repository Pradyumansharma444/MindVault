# 🔐 MindVault v2 — Privacy-First Personal Archivist

> Chat with your own writing. Surface hidden patterns. Visualise your thinking. **100% local. Zero cloud. Zero telemetry.**

---

## ✨ What's New in v2

| Feature | Description |
|---|---|
| **🕸 Concept Graph** | Interactive force-directed graph of keywords extracted from your archive via TF-IDF + NetworkX + Plotly |
| **🎭 Style Mimicry** | Toggle that rewrites the system prompt so the LLM responds *in your own writing voice* |
| **🔬 Verification Gate** | Cosine similarity check between answer and sources — answers below 0.45 are flagged `⚠ LOW CONFIDENCE` |
| **💬 Enhanced Chat** | Confidence badge on every response + source-file chips |
| **🔮 Echo** | 5-section pattern recognition report with contradictions, forgotten ideas, emotional arc |
| **🗂 Explorer** | Browse files with keyword chips extracted during ingestion |

---

## 🏗 Architecture

```
User Query
    │
    ▼
[ChromaDB MMR Retriever] ──── top-5 chunks ────▶ [RetrievalQA Chain]
    │                                                     │
    │                                              [Ollama LLM]
    │                                                     │
    │                                              [LLM Answer]
    │                                                     │
    └──── source chunks ──────▶ [Verification Gate] ──── cosine_similarity(answer, chunks)
                                       │
                              score ≥ 0.45? ──── ✓ VERIFIED
                                       │
                              score < 0.45? ──── ⚠ LOW CONFIDENCE

Concept Graph:
ChromaDB metadata["keywords"] ──▶ TF-IDF keywords ──▶ Co-occurrence graph
                                                            │
                                                     NetworkX layout ──▶ Plotly scatter
```

---

## 🚀 Setup — Step by Step

### Prerequisites

| Tool | Version | Where |
|---|---|---|
| Python | 3.11+ | [python.org](https://python.org) |
| Ollama | Latest | [ollama.com](https://ollama.com) |

---

### Step 1 — Install Ollama + Pull Model

**macOS / Linux:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# In Terminal 1 — keep open:
ollama serve

# In Terminal 2:
ollama pull llama3.2       # ~2 GB — recommended
# OR
ollama pull mistral        # faster on older CPUs

ollama list                # confirm download
```

**Windows (PowerShell as Admin):**
```powershell
# Download and run installer from https://ollama.com/download/windows
# Then in a new PowerShell window:
ollama serve

# New window:
ollama pull llama3.2
ollama list
```

---

### Step 2 — Python Environment

**macOS / Linux:**
```bash
cd mindvault_v2

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Windows:**
```powershell
cd mindvault_v2

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# If execution policy error:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 3 — Create Custom Ollama Model (Strongly Recommended)

This creates the `mindvault` persona with calibrated temperature and the archivist system prompt baked in.

```bash
ollama create mindvault -f Modelfile
ollama list    # should show: mindvault:latest
```

The app automatically detects and uses this model. If not found, it falls back to `llama3.2`.

---

### Step 4 — Index Your Documents

```bash
# Basic indexing:
python ingest.py --folder ~/Documents/Archive

# Different model:
python ingest.py --folder ~/Documents/Archive --model mistral

# Full re-index (wipes ChromaDB first):
python ingest.py --folder ~/Documents/Archive --reset
```

**Supported file types:** `.txt` `.md` `.pdf` `.docx` `.eml`

**What the ingestion does:**
1. Discovers all supported files recursively
2. Loads each file with the appropriate LangChain loader
3. Chunks with `RecursiveCharacterTextSplitter` (1000 chars, 200 overlap)
4. Extracts TF-IDF keywords per chunk (stored as metadata for the Concept Graph)
5. Embeds all chunks via Ollama and stores in ChromaDB

---

### Step 5 — Launch

```bash
streamlit run app.py
```

Browser opens at `http://localhost:8501` automatically.

---

## 🔬 Feature Deep-Dives

### Verification Gate

Every LLM answer is run through a cosine similarity check against its retrieved source chunks:

```
score = max(cosine_similarity(embed(answer), embed(source_chunk_i)))
```

| Score | Badge | Meaning |
|---|---|---|
| ≥ 0.70 | `✓ VERIFIED` (teal) | Answer closely mirrors source text |
| 0.45–0.70 | `◈ MEDIUM CONFIDENCE` (amber) | Reasonable grounding |
| < 0.45 | `⚠ LOW CONFIDENCE` (red) | Potential hallucination — treat with caution |

**Why this works:** If the LLM fabricates information not in the retrieved chunks, its answer will have low cosine similarity to those chunks when embedded. The same embedding model used for retrieval is used for verification — no extra model needed.

---

### Style Mimicry

Toggle in the sidebar. When **ON**, the chain is rebuilt with a completely different system prompt:

- **Normal mode:** "You are MindVault, a personal archivist AI. Cite sources. Say 'I don't see that' if not found."
- **Mimicry mode:** "You are the author. Respond ONLY in the writing style, vocabulary, sentence rhythm, and tone found in the context. Sound like the author, not an assistant."

The Verification Gate still runs in mimicry mode. If the model drifts into generic AI responses (low confidence), the badge will catch it.

**Use cases:**
- Ghost-writing drafts in your own voice
- Exploring how you'd respond to a question based on your past thinking
- Detecting if your writing style has shifted over time

---

### Concept Graph

Built from TF-IDF keyword metadata stored in ChromaDB during ingestion:

1. `ingest.py` runs `TfidfVectorizer` across all chunks, extracting top 8 keywords per chunk
2. Keywords stored as `metadata["keywords"]` = `"caffeine, morning routine, sleep, productivity, …"`
3. In the Graph tab, keywords become **nodes**, co-occurrence (same chunk) becomes **edges**
4. NetworkX `spring_layout` (Fruchterman–Reingold) computes x,y positions
5. Plotly renders as interactive scatter: hover for file provenance, scroll to zoom, drag to pan

**Colour scale:** grey → amber → teal = low → high connectivity

---

## 📦 PyInstaller Packaging

### macOS — Create `.app`

```bash
# Install PyInstaller
pip install pyinstaller

# Build (from mindvault_v2/ with venv active):
pyinstaller \
  --name MindVault \
  --onedir \
  --windowed \
  --osx-bundle-identifier com.mindvault.app \
  --hidden-import langchain \
  --hidden-import langchain_community \
  --hidden-import langchain_ollama \
  --hidden-import langchain_chroma \
  --hidden-import chromadb \
  --hidden-import networkx \
  --hidden-import plotly \
  --hidden-import sklearn \
  --hidden-import sklearn.metrics.pairwise \
  --hidden-import sklearn.feature_extraction.text \
  --hidden-import numpy \
  --add-data "app.py:." \
  --add-data "ingest.py:." \
  --add-data "Modelfile:." \
  launcher.py

# Output: dist/MindVault.app
open dist/MindVault.app
```

### Windows — Create `.exe`

```powershell
pip install pyinstaller

pyinstaller `
  --name MindVault `
  --onedir `
  --windowed `
  --hidden-import langchain `
  --hidden-import langchain_community `
  --hidden-import langchain_ollama `
  --hidden-import langchain_chroma `
  --hidden-import chromadb `
  --hidden-import networkx `
  --hidden-import plotly `
  --hidden-import sklearn `
  --hidden-import sklearn.metrics.pairwise `
  --hidden-import sklearn.feature_extraction.text `
  --hidden-import numpy `
  --add-data "app.py;." `
  --add-data "ingest.py;." `
  --add-data "Modelfile;." `
  launcher.py

# Output: dist\MindVault\MindVault.exe
```

### Linux — Create binary

```bash
pyinstaller \
  --name mindvault \
  --onedir \
  --hidden-import langchain \
  --hidden-import langchain_community \
  --hidden-import langchain_ollama \
  --hidden-import langchain_chroma \
  --hidden-import chromadb \
  --hidden-import networkx \
  --hidden-import plotly \
  --hidden-import sklearn \
  --hidden-import sklearn.metrics.pairwise \
  --hidden-import sklearn.feature_extraction.text \
  --hidden-import numpy \
  --add-data "app.py:." \
  --add-data "ingest.py:." \
  launcher.py

# Output: dist/mindvault/mindvault
chmod +x dist/mindvault/mindvault
./dist/mindvault/mindvault
```

> ⚠️ **Important:** The packaged app bundles Python + Streamlit. Ollama must be installed separately on the end user's machine — it cannot be bundled (it's a system service). `launcher.py` handles auto-starting `ollama serve` if not already running.

---

## ⚙️ Configuration Reference

Edit constants at the top of `ingest.py` and `app.py`:

| Constant | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `llama3.2` | Ollama model for embeddings |
| `LLM_MODEL` | `mindvault` | Ollama model for chat |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVER_K` | `5` | Chunks retrieved per query |
| `CONFIDENCE_THRESHOLD` | `0.45` | Verification Gate cutoff |
| `KEYWORDS_PER_CHUNK` | `8` | TF-IDF keywords stored per chunk |
| `MAX_GRAPH_NODES` | `80` | Max concept nodes in graph |

---

## 🛠 Troubleshooting

### "Ollama not running"
```bash
ollama serve     # Terminal 1 — keep open
```

### "Model not found"
```bash
ollama pull llama3.2
ollama list
```

### Concept graph is empty
Make sure you ran `ingest.py` from **this v2 version** — it stores `keywords` metadata. If you used the v1 ingestor, re-run with `--reset`.

### Verification gate always shows LOW CONFIDENCE
This can happen if:
- Your documents are very short or your query is very broad
- The embedding model hasn't warmed up yet (try a second query)
- Lower `CONFIDENCE_THRESHOLD` to `0.35` in `app.py`

### PDF loading errors
```bash
pip install pypdf --upgrade
# For scanned PDFs (OCR):
brew install tesseract     # macOS
pip install pytesseract
```

### ChromaDB migration error
```bash
# Nuke and re-index:
rm -rf chroma_db/
python ingest.py --folder ~/your/folder
```

### Streamlit port conflict
```bash
streamlit run app.py --server.port 8502
```

---

## 🔒 Privacy Guarantee

- **LLM inference:** Ollama runs entirely on your CPU/GPU. No API calls.
- **Vector storage:** ChromaDB persists to `./chroma_db/` — a plain local directory.
- **Keyword extraction:** Pure in-process scikit-learn. No network needed.
- **Streamlit telemetry:** Disabled via environment variable in `launcher.py`.
- **Verification:** Cosine similarity computed locally with numpy.
- **Zero external calls.** Verify with `Little Snitch` (macOS) or `Wireshark`.

---

## 🗺 Roadmap

- [ ] Incremental ingestion (skip already-indexed file hashes)
- [ ] Timeline view: keyword frequency over time as an area chart
- [ ] Voice input via local Whisper
- [ ] Semantic search (bypass LLM, return raw chunk matches)
- [ ] Export annotated concept graph as PNG/SVG

---

*Built with Python 3.11 · LangChain · ChromaDB · Ollama · Streamlit · NetworkX · Plotly · scikit-learn*
