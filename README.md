# Tndrr AI Internship Assignment

This repository contains the assignments and projects completed for the **Tndrr AI Internship**.

---

## 📁 Repository Structure

### 🔹 [`task-1/`](task-1/) — Parameter-Efficient Fine-Tuning (LoRA) for Structured JSON Parsing
* **Core Goal:** Transform a lightweight Small Language Model (SLM $\le 0.5\text{B}$) into a deterministic, schema-compliant JSON function-calling agent for autonomous workflows.
* **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (494M parameters, RoPE, SwiGLU, RMSNorm, GQA, ChatML).
* **Fine-Tuning Method:** Parameter-Efficient Fine-Tuning (PEFT) via **LoRA** ($r=16, \alpha=32$, targeting `q_proj, k_proj, v_proj, o_proj`) with strict prompt-loss masking.
* **Trainable Parameters:** **2,162,688 (0.4359%)** | Adapter file size: **8.67 MB** (`adapter_model.safetensors`).
* **Empirical Comparison (160 Held-Out Test Samples):**
  * **Intent Classification Accuracy:** **0.00% (Base)** $\rightarrow$ **98.75% (LoRA)** (**+98.75%**)
  * **Canonical Exact Match (EM):** **0.00% (Base)** $\rightarrow$ **64.38% (LoRA)** (**+64.38%**)
  * **Field Micro F1 Score:** **5.97% (Base)** $\rightarrow$ **86.75% (LoRA)** (**+80.78% / +1,353% rel**)
  * **Raw JSON Validity Rate:** **78.12% (Base)** $\rightarrow$ **99.38% (LoRA)** (**+21.26%**)
* **Key Components:**
  * Complete 13-section technical report: [`task-1/TASK_1_FINAL_REPORT.md`](task-1/TASK_1_FINAL_REPORT.md)
  * Theoretical 7B GPU memory estimation (Full FT vs LoRA vs QLoRA): [`task-1/reports/memory_calculation.md`](task-1/reports/memory_calculation.md)
  * Pretrained LoRA adapter weights: [`task-1/models/qwen2.5-0.5b-lora/best_adapter/`](task-1/models/qwen2.5-0.5b-lora/best_adapter/)
  * Turnkey interactive notebook (Google Colab / Kaggle ready): [`task-1/notebooks/finetuning_experiment.ipynb`](task-1/notebooks/finetuning_experiment.ipynb)
  * Dissected failure case studies with root causes & mitigations: [`task-1/results/`](task-1/results/)

---

### 🔹 [`task-2/`](task-2/) — Retrieval-Augmented Generation (RAG) From Scratch
* **Core Goal:** Build an end-to-end Question-Answering system over a public document of $\ge 30$ pages from scratch without LangChain, LlamaIndex, or Haystack.
* **Document Dataset:** NIST AI 100-1: *Artificial Intelligence Risk Management Framework (AI RMF 1.0)* (48 Pages, Public Domain).
* **Architecture:** Modular implementation using pure Python, `pypdf`, `sentence-transformers` (`all-MiniLM-L6-v2`), NumPy, FAISS `IndexFlatIP`, native BM25, and grounded prompt synthesis with page citations.
* **Empirical Comparison (15 Verified Questions):**
  * **Small Chunks (~250 words, 50-word overlap):** **86.67% Recall@5**, **26.67% Precision@5**, **0.6722 MRR** (Winner: High semantic specificity, sharp vector representations).
  * **Large Chunks (~500 words, 100-word overlap):** **80.00% Recall@5**, **18.67% Precision@5**, **0.5244 MRR** (Suffered from semantic dilution across adjacent topics).
* **Key Components:**
  * Complete 22-section academic report: [`task-2/TASK_2_FINAL_REPORT.md`](task-2/TASK_2_FINAL_REPORT.md)
  * Runnable interactive notebook (Google Colab ready): [`task-2/rag_notebook.ipynb`](task-2/rag_notebook.ipynb)
  * Two detailed retrieval failure case studies ("Table of Contents Distractor Trap" and "Figure Layout Fragmentation")
  * Theoretical exposition of Okapi BM25 and Cross-Encoder Re-Ranking with ranking inversion examples.

---

### 🔹 [`task-3/`](task-3/) — Running an Open LLM Locally & Optimization
* **Hardware:** NVIDIA GeForce RTX 2050 (4 GB VRAM) + AMD Ryzen 7 5800HS + 16 GB RAM.
* **Model:** Meta Llama 3.2 3B Instruct evaluated across 4-bit (`Q4_K_M`) vs. 8-bit (`Q8_0`) quantization.
* **Measured Performance:** **39.43 tokens/sec (4-bit)** vs. **14.18 tokens/sec (8-bit)** (2.78x speedup).
* **Key Components:**
  * Contains automated benchmarking harness, raw JSON telemetry, 10-prompt qualitative comparison, KV cache / quantization theory, and 20-user 24 GB GPU capacity plan.
  * See the full report in [`task-3/TASK_3_FINAL_REPORT.md`](task-3/TASK_3_FINAL_REPORT.md).
