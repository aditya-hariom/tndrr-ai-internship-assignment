"""
RAG from Scratch - NIST AI Risk Management Framework (AI RMF 1.0)
==================================================================
A transparent, framework-free Retrieval-Augmented Generation system.
Built using:
  - pypdf (document loading)
  - standard regex & string processing (text cleaning)
  - sliding window token/word chunking with metadata tracking
  - sentence-transformers (all-MiniLM-L6-v2) for embeddings
  - NumPy & FAISS for vector similarity search (cosine / inner product)
  - Native BM25 implementation from scratch for hybrid retrieval analysis
  - Grounded LLM generation with strict page citations
"""

import os
import re
import math
import json
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pypdf

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ============================================================================
# STAGE A: DOCUMENT LOADING
# ============================================================================

class DocumentLoader:
    """
    Loads PDF documents page by page, extracting text and preserving
    critical metadata (document name, page number, character count).
    Handles empty pages and extraction errors gracefully.
    """
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc_name = os.path.basename(pdf_path)

    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF document not found at: {self.pdf_path}")

        reader = pypdf.PdfReader(self.pdf_path)
        pages = []
        total_pages = len(reader.pages)

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            try:
                page_obj = reader.pages[page_idx]
                extracted_text = page_obj.extract_text() or ""
                pages.append({
                    "doc_name": self.doc_name,
                    "page_number": page_num,
                    "text": extracted_text,
                    "char_count": len(extracted_text),
                    "is_empty": len(extracted_text.strip()) == 0
                })
            except Exception as e:
                print(f"[Warning] Error extracting page {page_num}: {e}")
                pages.append({
                    "doc_name": self.doc_name,
                    "page_number": page_num,
                    "text": "",
                    "char_count": 0,
                    "is_empty": True,
                    "error": str(e)
                })

        return pages


# ============================================================================
# STAGE B: TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """
    Cleans raw PDF text while preserving semantic structure:
    - Normalizes multiple spaces, tabs, and inconsistent linebreaks
    - Merges hyphenated words split across line breaks (e.g. 'instan-tiation' -> 'instantiation')
    - Removes repeated running headers/footers (e.g. 'NIST AI 100-1 AI RMF 1.0')
    - Preserves meaningful section headings, bullets, and paragraph boundaries
    """
    HEADER_PATTERN = re.compile(r"NIST\s+AI\s+100-1\s+(AI\s+RMF\s+1\.0)?", re.IGNORECASE)
    PAGE_NUMBER_FOOTER = re.compile(r"\n\s*\d+\s*$", re.MULTILINE)
    HYPHENATION_PATTERN = re.compile(r"(\b\w+)-\n(\w+\b)")

    @classmethod
    def clean_text(cls, text: str) -> str:
        if not text:
            return ""

        # Remove running headers
        cleaned = cls.HEADER_PATTERN.sub("", text)
        
        # Merge hyphenated words across lines
        cleaned = cls.HYPHENATION_PATTERN.sub(r"\1\2", cleaned)

        # Replace carriage returns
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")

        # Replace excessive whitespace within lines
        lines = [line.strip() for line in cleaned.split("\n")]
        
        # Filter out isolated page number lines
        filtered_lines = [l for l in lines if not re.match(r"^\d{1,3}$", l)]

        # Join lines into coherent paragraphs
        joined = "\n".join(filtered_lines)
        
        # Collapse 3+ newlines to 2 newlines (preserve paragraph separation)
        joined = re.sub(r"\n{3,}", "\n\n", joined)
        
        # Collapse multiple spaces to single space
        joined = re.sub(r"[ \t]+", " ", joined)

        return joined.strip()


# ============================================================================
# STAGE C: CHUNKING
# ============================================================================

class Chunker:
    """
    Implements sliding window chunking with configurable chunk size
    and overlap. Preserves cross-page boundaries and rich chunk metadata.
    """
    def __init__(self, chunk_size_words: int, overlap_words: int, config_name: str):
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words
        self.config_name = config_name
        self.step_size = max(1, chunk_size_words - overlap_words)

    def chunk_document(self, cleaned_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates chunks from page-structured text while recording
        all source pages spanned by each chunk.
        """
        # Build token/word stream with page tracking for each word
        word_stream = []
        for p in cleaned_pages:
            words = p["cleaned_text"].split()
            for w in words:
                word_stream.append((w, p["page_number"], p["doc_name"]))

        if not word_stream:
            return []

        chunks = []
        chunk_idx = 0
        total_words = len(word_stream)

        start_idx = 0
        while start_idx < total_words:
            end_idx = min(start_idx + self.chunk_size_words, total_words)
            chunk_slice = word_stream[start_idx:end_idx]

            chunk_text = " ".join([w[0] for w in chunk_slice])
            pages_in_chunk = sorted(list(set([w[1] for w in chunk_slice])))
            primary_page = pages_in_chunk[0]
            doc_name = chunk_slice[0][2]

            chunk_id = f"{self.config_name}_chunk_{chunk_idx + 1:04d}"

            chunks.append({
                "chunk_id": chunk_id,
                "config": self.config_name,
                "chunk_index": chunk_idx,
                "doc_name": doc_name,
                "primary_page": primary_page,
                "page_numbers": pages_in_chunk,
                "word_count": len(chunk_slice),
                "char_count": len(chunk_text),
                "text": chunk_text
            })

            chunk_idx += 1
            if end_idx >= total_words:
                break
            start_idx += self.step_size

        return chunks


# ============================================================================
# STAGE D: EMBEDDINGS
# ============================================================================

class EmbeddingEngine:
    """
    Wraps an open-source SentenceTransformer embedding model.
    Encodes text into dense semantic vector representations.
    Applies L2 normalization so cosine similarity is computed
    directly via inner (dot) product.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        print(f"[EmbeddingEngine] Loading model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"[EmbeddingEngine] Model loaded. Embedding dimension = {self.embedding_dim}")

    def embed_texts(self, texts: List[str], normalize: bool = True, batch_size: int = 32) -> np.ndarray:
        """
        Embeds a list of strings into a float32 NumPy array (N, D).
        If normalize=True, vectors are scaled to unit length ||v||_2 = 1.
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str, normalize: bool = True) -> np.ndarray:
        """
        Embeds a single query string into a 1D or 2D float32 vector.
        """
        vec = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        return vec.astype(np.float32)


# ============================================================================
# STAGE E: VECTOR DATABASE / RETRIEVAL
# ============================================================================

class VectorIndex:
    """
    Vector similarity search index supporting both:
      1. FAISS IndexFlatIP (Inner Product over normalized vectors = Cosine Similarity)
      2. Pure NumPy implementation using matrix dot product (no binary dependencies)
    """
    def __init__(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]], use_faiss: bool = True):
        self.chunks = chunks
        self.embeddings = embeddings  # (N, D) float32 normalized
        self.dim = embeddings.shape[1]
        self.num_chunks = embeddings.shape[0]
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        if self.use_faiss:
            # IndexFlatIP computes exact inner product
            self.index = faiss.IndexFlatIP(self.dim)
            self.index.add(self.embeddings)
            print(f"[VectorIndex] Built FAISS IndexFlatIP with {self.index.ntotal} vectors.")
        else:
            self.index = None
            print(f"[VectorIndex] Built pure NumPy vector index with {self.num_chunks} vectors.")

    def search_faiss(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """Searches using FAISS."""
        distances, indices = self.index.search(query_vec, top_k)
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(distances[0][rank])))
        return results

    def search_numpy(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Pure NumPy Cosine Similarity:
          S = Q . V^T
        Because both query_vec and database vectors are L2-normalized:
          dot_product(q, v) = cos(theta)
        """
        # query_vec is (1, D), self.embeddings is (N, D)
        scores = np.dot(self.embeddings, query_vec.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results

    def retrieve(self, query_vec: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        if self.use_faiss:
            return self.search_faiss(query_vec, top_k=top_k)
        else:
            return self.search_numpy(query_vec, top_k=top_k)


# ============================================================================
# BM25 KEYWORD RETRIEVER (From Scratch for Comparative Analysis)
# ============================================================================

class BM25Retriever:
    """
    Okapi BM25 implementation from scratch using standard Python math.
    Parameters:
      k1: term frequency saturation factor (default 1.5)
      b: document length normalization factor (default 0.75)
    """
    def __init__(self, chunks: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.N = len(chunks)
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.idf = {}
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _build_index(self):
        doc_freqs = {}
        total_len = 0

        for chunk in self.chunks:
            tokens = self._tokenize(chunk["text"])
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_len += doc_len

            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.doc_term_freqs.append(tf)

            for t in tf.keys():
                doc_freqs[t] = doc_freqs.get(t, 0) + 1

        self.avg_doc_len = total_len / max(1, self.N)

        # Standard Lucene/BM25 IDF formula with smoothing
        for term, df in doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.N, dtype=np.float32)

        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]

            for doc_idx, tf_dict in enumerate(self.doc_term_freqs):
                if token in tf_dict:
                    freq = tf_dict[token]
                    doc_len = self.doc_lengths[doc_idx]
                    # BM25 term score formula
                    numerator = freq * (self.k1 + 1.0)
                    denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                    scores[doc_idx] += idf_val * (numerator / denominator)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], float(scores[idx])))
        return results


# ============================================================================
# STAGE F: ANSWER GENERATION
# ============================================================================

PROMPT_TEMPLATE = """You are an authoritative AI assistant providing factual answers strictly based on the retrieved context passages from the National Institute of Standards and Technology (NIST) Artificial Intelligence Risk Management Framework (AI RMF 1.0).

STRICT INSTRUCTIONS:
1. Answer the question using ONLY the facts directly stated in the context passages below.
2. Do NOT invent, assume, extrapolate, or bring in external knowledge.
3. If the retrieved context does not contain sufficient facts to answer the question, state: "I could not find sufficient information in the provided documents."
4. Every statement or claim in your response MUST cite its source document and page number in the format: [Source: <doc_name>, Page <page_number>].

---
RETRIEVED CONTEXT PASSAGES:
{context_block}
---

QUESTION:
{question}

FACTUAL ANSWER (with page citations):"""


class LLMGenerator:
    """
    Handles prompt construction and LLM answer generation:
    - Formats retrieved chunks into indexed context blocks with page numbers
    - Connects to Google Gemini API if GEMINI_API_KEY is available
    - Connects to OpenAI API if OPENAI_API_KEY is available
    - Provides a grounded extractive generator as an offline fallback
      so the system always produces an answer and citations without crashing.
    """
    def __init__(self, api_provider: str = "auto", model_name: str = "gemini-2.5-flash"):
        self.api_provider = api_provider
        self.model_name = model_name
        self.gemini_client = None
        self.openai_client = None

        gemini_key = os.environ.get("GEMINI_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")

        if (api_provider in ("auto", "gemini")) and gemini_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=gemini_key)
                self.active_provider = "gemini"
                print(f"[LLMGenerator] Active provider: Google Gemini ({model_name})")
            except Exception as e:
                print(f"[LLMGenerator] Could not initialize Gemini: {e}")

        elif (api_provider in ("auto", "openai")) and openai_key:
            try:
                import openai
                self.openai_client = openai.OpenAI(api_key=openai_key)
                self.active_provider = "openai"
                print("[LLMGenerator] Active provider: OpenAI")
            except Exception as e:
                print(f"[LLMGenerator] Could not initialize OpenAI: {e}")

        if not self.gemini_client and not self.openai_client:
            self.active_provider = "grounded_fallback"
            print("[LLMGenerator] No API key detected. Using deterministic grounded fallback generator.")

    def format_prompt(self, question: str, retrieved_chunks: List[Tuple[Dict[str, Any], float]]) -> str:
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved_chunks):
            doc_name = chunk.get("doc_name", "NIST_AI_RMF_1.0.pdf")
            pages_str = ", ".join(str(p) for p in chunk.get("page_numbers", [chunk.get("primary_page")]))
            chunk_id = chunk.get("chunk_id", f"chunk_{i+1}")
            context_parts.append(
                f"[Passage {i+1}] (Chunk ID: {chunk_id} | Document: {doc_name} | Page: {pages_str} | Similarity: {score:.4f}):\n{chunk['text']}"
            )

        context_block = "\n\n".join(context_parts)
        return PROMPT_TEMPLATE.format(context_block=context_block, question=question)

    def generate_answer(self, question: str, retrieved_chunks: List[Tuple[Dict[str, Any], float]]) -> Dict[str, Any]:
        prompt = self.format_prompt(question, retrieved_chunks)

        if self.active_provider == "gemini" and self.gemini_client:
            try:
                response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                answer_text = response.text.strip()
                return {"answer": answer_text, "provider": "gemini", "prompt": prompt}
            except Exception as e:
                print(f"[LLMGenerator] Gemini call failed: {e}. Falling back.")

        # Grounded Fallback Answer Generation (offline safe)
        top_chunk, top_score = retrieved_chunks[0]
        primary_page = top_chunk.get("primary_page", "Unknown")
        doc_name = top_chunk.get("doc_name", "NIST_AI_RMF_1.0.pdf")
        
        # Take the most relevant sentences from top retrieved chunk
        sentences = re.split(r"(?<=[.!?])\s+", top_chunk["text"])
        snippet = " ".join(sentences[:3]) if len(sentences) >= 3 else top_chunk["text"]

        fallback_answer = (
            f"Based on the retrieved source passages, {snippet} "
            f"[Source: {doc_name}, Page {primary_page}]."
        )
        return {"answer": fallback_answer, "provider": "grounded_fallback", "prompt": prompt}


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_retrieval(
    retrieved_chunks: List[Tuple[Dict[str, Any], float]],
    target_page: int,
    ground_truth_passage: str,
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Evaluates whether the relevant information was successfully retrieved in top-k.
    Calculates:
      - is_hit: bool (Recall@k = 1 if hit, 0 if not)
      - hit_rank: rank (1 to k) of first relevant chunk, or None
      - reciprocal_rank: 1 / hit_rank
      - precision_at_k: count of relevant chunks / k
    """
    # Key tokens from ground truth passage
    gt_tokens = set([t.lower() for t in re.findall(r"\b\w{4,}\b", ground_truth_passage)])

    hit = False
    hit_rank = None
    relevant_count = 0

    for rank, (chunk, score) in enumerate(retrieved_chunks[:top_k], start=1):
        chunk_pages = chunk.get("page_numbers", [chunk.get("primary_page")])
        chunk_text = chunk.get("text", "").lower()

        # Check page match
        page_match = target_page in chunk_pages

        # Check lexical overlap with ground truth passage
        chunk_tokens = set(re.findall(r"\b\w{4,}\b", chunk_text))
        overlap = len(gt_tokens.intersection(chunk_tokens)) / max(1, len(gt_tokens))

        # A chunk is considered relevant if it covers the target page AND has substantive overlap
        is_relevant = page_match and (overlap >= 0.25 or any(phrase.lower() in chunk_text for phrase in ground_truth_passage.split(";")[:2]))

        if is_relevant:
            relevant_count += 1
            if not hit:
                hit = True
                hit_rank = rank

    reciprocal_rank = 1.0 / hit_rank if hit else 0.0
    precision = relevant_count / float(top_k)

    return {
        "recall_at_k": 1.0 if hit else 0.0,
        "is_hit": hit,
        "hit_rank": hit_rank,
        "reciprocal_rank": reciprocal_rank,
        "precision_at_k": precision,
        "relevant_count": relevant_count
    }

