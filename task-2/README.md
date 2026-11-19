# Retrieval-Augmented Generation (RAG) From Scratch

An academic, transparent implementation of a Question-Answering system using Retrieval-Augmented Generation (RAG) from scratch over public documents.

> **Zero Framework Guarantee:** This project does **NOT** use LangChain, LlamaIndex, Haystack, or any similar orchestration wrapper. All components—document loading, text cleaning, sliding window chunking, embedding generation, vector similarity search, BM25 retrieval, prompt structuring, and evaluation metrics—are implemented manually using pure Python, NumPy, FAISS, and PyTorch/Sentence-Transformers.

---

## 1. Selected Public Document Dataset

- **Document Name:** NIST AI 100-1: *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*
- **Authoring Organization:** National Institute of Standards and Technology (NIST), U.S. Department of Commerce
- **Publication Date:** January 2023
- **Page Count:** 48 pages (meets the $\ge 30$ pages requirement)
- **Direct Official Download Link:** [https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf)
- **Copyright & License Status:** United States Government Work (Public Domain, Title 17 U.S.C. § 105), freely distributable and ideal for academic reproduction without licensing restrictions.

---

## 2. Project File Structure

```
d:/Ai/Task2/
├── download_dataset.py       # Automated script to fetch NIST AI RMF 1.0 PDF
├── rag_pipeline.py           # Core RAG classes: Loader, Preprocessor, Chunker,
│                             # EmbeddingEngine, VectorIndex, BM25, LLMGenerator
├── evaluate_rag.py           # Evaluation runner: 15 questions, Small vs Large chunking,
│                             # Recall@5, Precision@5, MRR calculation, failure logs
├── inspect_failures.py       # Deep-dive diagnostic tool for retrieval error analysis
├── rag_notebook.ipynb        # Standalone, interactive Google Colab / Jupyter notebook
├── evaluation_results.json   # Full raw experimental results, chunk rankings, and metrics
├── NIST_AI_RMF_1.0.pdf       # The 48-page target dataset PDF
├── README.md                 # Setup, architecture, and execution guide
└── academic_report.md        # Comprehensive 22-section academic assignment report
```

---

## 3. Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12 (CPython 64-bit recommended)
- `pip` or `uv` package manager

### Local Environment Setup
```bash
# Clone or navigate to the project directory
cd d:/Ai/Task2

# Create a virtual environment (e.g. using Python 3.12 or uv)
python -m venv .venv

# Activate the virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Linux/macOS:
# source .venv/bin/activate

# Install required dependencies
pip install pypdf sentence-transformers faiss-cpu numpy requests tabulate google-genai
```

### LLM API Key Configuration (Optional)
The system supports LLM answer generation via:
1. **Google Gemini API:** Set `GEMINI_API_KEY`
2. **OpenAI API:** Set `OPENAI_API_KEY`
3. **Deterministic Grounded Fallback:** If no API key is provided, the pipeline runs an offline grounded generator that extracts top retrieved passages and synthesizes strict page citations without crashing.

To set your Gemini API key:
```powershell
$env:GEMINI_API_KEY = "your_actual_api_key_here"
```

---

## 4. Execution Commands

### 1. Download Document
```bash
python download_dataset.py
```

### 2. Run Complete RAG Pipeline & Evaluation
```bash
python evaluate_rag.py
```
This script will:
- Parse all 48 pages of `NIST_AI_RMF_1.0.pdf`.
- Clean headers, hyphens, and whitespace.
- Partition text into **Small Chunks (~250 words)** and **Large Chunks (~500 words)**.
- Encode chunks using `sentence-transformers/all-MiniLM-L6-v2`.
- Build vector indexes with FAISS `IndexFlatIP` and NumPy cosine search.
- Evaluate the 15 original questions against both configurations.
- Compute and print Recall@5, Precision@5, and Mean Reciprocal Rank (MRR).
- Save results to `evaluation_results.json`.

### 3. Analyze Retrieval Failures
```bash
python inspect_failures.py
```

### 4. Interactive Google Colab / Jupyter Notebook
Launch JupyterLab or Google Colab and open `rag_notebook.ipynb`. Every cell is fully commented and runnable from top to bottom.

---

## 5. Summary of Experimental Results

| Metric | Small Chunks (~250 words, 50 overlap) | Large Chunks (~500 words, 100 overlap) |
|---|---:|---:|
| **Total Chunks Generated** | 77 chunks | 39 chunks |
| **Recall@5** | **86.67%** (13/15 hits) | 80.00% (12/15 hits) |
| **Precision@5** | **26.67%** | 18.67% |
| **Mean Reciprocal Rank (MRR)** | **0.6722** | 0.5244 |
| **Primary Strength** | Pinpoint semantic accuracy for specific definitions | Context preservation spanning multiple paragraphs |

