# Reproducible LoRA Fine-Tuning: Natural Language to Structured JSON Action Parsing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg)](https://pytorch.org/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA-FF6F00.svg)](https://github.com/huggingface/peft)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade, fully reproducible parameter-efficient fine-tuning project converting natural language user utterances into strictly typed, validated JSON function/action calls using **LoRA (Low-Rank Adaptation)** on **`Qwen/Qwen2.5-0.5B-Instruct`**.

---

## Table of Contents
1. [Project Overview & Problem Statement](#1-project-overview--problem-statement)
2. [Why This Task Was Selected](#2-why-this-task-was-selected)
3. [Base Model Profile](#3-base-model-profile)
4. [Predefined JSON Schema & Intent Taxonomy](#4-predefined-json-schema--intent-taxonomy)
5. [Dataset Engineering & Leakage Prevention](#5-dataset-engineering--leakage-prevention)
6. [Training Methodology & LoRA Configuration](#6-training-methodology--lora-configuration)
7. [Evaluation Methodology & Mathematical Metrics](#7-evaluation-methodology--mathematical-metrics)
8. [Experimental Results: Base vs. Fine-Tuned](#8-experimental-results-base-vs-fine-tuned)
9. [Qualitative Failure Analysis](#9-qualitative-failure-analysis)
10. [7B Parameter Model GPU Memory Calculation](#10-7b-parameter-model-gpu-memory-calculation)
11. [How to Reproduce the Experiment](#11-how-to-reproduce-the-experiment)
12. [Hardware & Software Specifications](#12-hardware--software-specifications)
13. [Limitations & Future Roadmap](#13-limitations--future-roadmap)

---

## 1. Project Overview & Problem Statement

Language models running as autonomous agents or edge microservices must reliably convert unstructured, conversational human language into deterministic, machine-readable JSON actions. General-purpose instruction-tuned models frequently fail in production settings:
1. They hallucinate arbitrary keys (e.g., emitting `"action"` instead of `"intent"`).
2. They wrap outputs in conversational prose or markdown fences (````json ... ````).
3. They hallucinate default values when optional fields are unspecified rather than emitting `null`.
4. They fail numeric conversions (e.g., mapping `"15 minutes"` to `duration_seconds: 900`).

This project demonstrates how **Parameter-Efficient Fine-Tuning (PEFT)** via **LoRA** transforms a lightweight 494M parameter base model into a deterministic JSON action engine, training **only 0.4359% of weights (2.16M parameters / 8.67 MB)** while boosting Intent Accuracy from **0.00% to 98.75%** and Field Micro F1 from **5.97% to 86.75%**.

---

## 2. Why This Task Was Selected

* **High Commercial Relevance:** Function calling and semantic slot extraction are foundational to modern AI agent frameworks, mobile on-device assistants, and API gateways.
* **Rigorous Evaluation Criteria:** Unlike open-ended generation, structured JSON allows strict, objective mathematical metrics (exact schema conformance, canonical JSON equality, per-slot precision/recall).
* **SLM Optimization:** Demonstrates that a small language model (SLM $\le 0.5\text{B}$) can match or exceed large models on specialized routing tasks at a fraction of the inference latency and zero API cost.

---

## 3. Base Model Profile

* **Identifier:** `Qwen/Qwen2.5-0.5B-Instruct`
* **Architecture:** Qwen2ForCausalLM (RoPE positional embeddings, SwiGLU activations, RMSNorm, Grouped Query Attention)
* **Total Parameters:** 494,032,768 parameters
* **Native Context Length:** 32,768 tokens (configured to 256 for task efficiency)
* **Vocabulary Size:** 151,936 tokens
* **Native Template:** ChatML format (`<|im_start|>system...<|im_end|>\n<|im_start|>user...<|im_end|>\n<|im_start|>assistant\n`)

---

## 4. Predefined JSON Schema & Intent Taxonomy

The dataset and model enforce a strict 4-intent schema plus a fallback:

```json
{
  "create_reminder": {
    "required": ["intent", "task"],
    "optional": ["date", "time", "priority"]
  },
  "schedule_meeting": {
    "required": ["intent", "title", "participants"],
    "optional": ["date", "time", "duration_minutes"]
  },
  "set_timer": {
    "required": ["intent", "duration_seconds"],
    "optional": ["label"]
  },
  "log_expense": {
    "required": ["intent", "amount", "currency"],
    "optional": ["category", "merchant"]
  },
  "unknown": {
    "required": ["intent", "error", "raw_query"]
  }
}
```

* **Missing Slot Rule:** Any unmentioned optional field **must be emitted as `null`**, never omitted and never populated with hallucinated defaults.

---

## 5. Dataset Engineering & Leakage Prevention

### 5.1 Dataset Split & Stratification
A total of **1,600 examples** were generated with a fixed random seed (`seed=42`):
* **Training Set (`data/train.jsonl`):** 1,280 samples (80.0%)
* **Validation Set (`data/validation.jsonl`):** 160 samples (10.0%)
* **Held-Out Test Set (`data/test.jsonl`):** 160 samples (10.0%) — *Exceeds the $\ge 50$ requirement*

### 5.2 Zero-Leakage Architecture
To ensure genuine generalization rather than verbatim memorization:
1. **Global Deduplication:** A global hash set enforces **zero exact query collisions** across `train`, `validation`, and `test` splits ($0$ overlapping queries confirmed by audit).
2. **Disjoint Template Families:** Distinct linguistic structures are assigned to eval splits (direct commands in train vs polite/elliptical/terse phrasing in test).
3. **Disjoint Entity Sets:** Named entities, merchants, tasks, and meeting titles are partitioned into separate disjoint subsets for train vs eval splits.
4. **Lexical Isolation:** Cross-split 4-gram Jaccard overlap is $0.0004$ (train/val) and $0.0089$ (train/test).

---

## 6. Training Methodology & LoRA Configuration

### 6.1 LoRA Configuration & Hyperparameter Justification

| Hyperparameter | Value | Technical Justification |
| :--- | :--- | :--- |
| **LoRA Rank ($r$)** | `16` | Provides sufficient rank-space capacity for nonlinear slot mapping while keeping trainable parameters under 0.5%. |
| **LoRA Alpha ($\alpha$)** | `32` | Standard scaling ratio $\frac{\alpha}{r} = 2.0$, ensuring adapter gradient updates match the magnitude of base weight gradients. |
| **LoRA Dropout** | `0.05` | Regularizes adapter activations against memorizing specific template tokens. |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj` | Adapts all 4 attention projection matrices, capturing token routing and cross-slot contextual associations. |
| **Learning Rate** | $2.0 \times 10^{-4}$ | Standard LoRA learning rate with AdamW, enabling fast convergence without catastrophic forgetting. |
| **Weight Decay** | `0.01` | L2 weight regularization preventing adapter parameter explosion. |
| **Batch Size & Grad Accum** | Batch `2`, Accum `4` | **Effective batch size = 8** ($2 \times 4$), maintaining low VRAM while providing stable gradient updates. |
| **Warmup Ratio** | `0.05` | Linear warmup over first 24 steps prevents initial gradient instability. |
| **LR Scheduler** | Cosine Decay | Smoothly anneals learning rate to 0 over 480 total optimizer steps. |
| **Precision** | `bfloat16` | Native 16-bit brain floating point supported by Ampere architecture. |

### 6.2 Prompt Loss Masking
Standard causal language modeling computes cross-entropy over all tokens. In this implementation:
* Tokens in the System Prompt and User Prompt are masked with `labels = -100` (`IGNORE_INDEX`).
* Cross-entropy loss is computed **strictly over the target assistant JSON tokens**, preventing the model from wasting parameter capacity on learning to reproduce input prompts.

### 6.3 Parameter Statistics
* **Base Model Parameters:** 494,032,768
* **Trainable Parameters:** **2,162,688 (0.4359%)**
* **Frozen Parameters:** 494,032,768 (99.5641%)
* **Adapter File Size:** **8.67 MB** (`adapter_model.safetensors`)

---

## 7. Evaluation Methodology & Mathematical Metrics

Evaluation was conducted on all 160 held-out test examples under identical deterministic generation settings (`do_sample=False`, `temperature=0.0`, greedy decoding).

1. **JSON Validity Rate ($\text{JVR}$):** Proportion of outputs that successfully parse via `json.loads()`.
2. **Intent Classification Accuracy ($\text{ICA}$):** Proportion of predictions where `pred["intent"] == ground_truth["intent"]`.
3. **Exact Match (EM) Accuracy ($\text{EMA}$):** Canonical recursive equivalence across all keys and values (whitespace-invariant, key-order-invariant).
4. **Field Micro F1 Score:** Harmonic mean of precision and recall over all slot assignments (excluding `intent`).

---

## 8. Experimental Results: Base vs. Fine-Tuned

*Actual experimental measurements on the 160 held-out test examples:*

| Evaluation Metric | Untuned Base Model | LoRA Fine-Tuned Model | Absolute Improvement ($\Delta$) | Relative Improvement (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Raw JSON Validity Rate** | 78.12% | **99.38%** | **+21.26%** | **+27.2%** |
| **Extracted JSON Validity** | 85.62% | **99.38%** | **+13.76%** | **+16.1%** |
| **Intent Classification Accuracy** | 0.00% | **98.75%** | **+98.75%** | *N/A (base was 0%)* |
| **Exact Match (EM) Accuracy** | 0.00% | **64.38%** | **+64.38%** | *N/A (base was 0%)* |
| **Field Micro Precision** | 8.28% | **86.98%** | **+78.70%** | **+950.5%** |
| **Field Micro Recall** | 4.66% | **86.53%** | **+81.87%** | **+1,756.9%** |
| **Field Micro F1 Score** | 5.97% | **86.75%** | **+80.78%** | **+1,353.1%** |
| **Field Micro Accuracy** | 3.08% | **76.61%** | **+73.53%** | **+2,387.3%** |

```
=============================================================================================
                                  PERFORMANCE COMPARISON
=============================================================================================
Intent Accuracy       Base:  0.0%  
                      LoRA: [=================================================] 98.8% (+98.8%)

Field Micro F1        Base: [===] 6.0%
                      LoRA: [===========================================] 86.8% (+80.8%)

Exact Match (EM)      Base:  0.0%
                      LoRA: [=================================] 64.4% (+64.4%)

Raw JSON Validity     Base: [=======================================] 78.1%
                      LoRA: [==================================================] 99.4% (+21.3%)
=============================================================================================
```

---

## 9. Qualitative Failure Analysis

Extracted from real test predictions in `results/finetuned_predictions.jsonl`:

### Failure Case 1: Out-of-Schema Intent Hallucination
* **Sample ID:** `test_0018`
* **Input Query:** `"Track twenty dollars."`
* **Expected JSON:**
  ```json
  {"intent": "unknown", "error": "insufficient_information", "raw_query": "Track twenty dollars."}
  ```
* **Actual Output:**
  ```json
  {"intent": "set_goal", "amount": 20.0, "category": null, "label": "Track"}
  ```
* **Category:** Ambiguous Request / Out-of-Schema Hallucination
* **Root Cause:** The query lacks direction (expense vs goal) and merchant. The model attempted to be overly helpful, hallucinating a non-existent intent `"set_goal"`.
* **Mitigation:** Expand ambiguous training examples mapped to `unknown` and implement grammar-constrained decoding (e.g. Outlines/regex logits processor).

### Failure Case 2: Entity Span Boundary Bleed (Over-Extraction)
* **Sample ID:** `test_0006`
* **Input Query:** `"Set up a Customer advisory council discussion with Kavita and Amara on on Jan 15th at 10:00 AM for 90-minute."`
* **Expected JSON:**
  ```json
  {"intent": "schedule_meeting", "title": "Customer advisory council", "participants": ["Kavita", "Amara"], "date": "on Jan 15th", "time": "10:00", "duration_minutes": 90}
  ```
* **Actual Output:**
  ```json
  {"intent": "schedule_meeting", "title": "Customer advisory council discussion", "participants": ["Kavita", "Amara"], "date": "on Jan 15th", "time": "10:00", "duration_minutes": 90}
  ```
* **Category:** Entity Span Boundary Delineation
* **Root Cause:** Absorbed the structural noun `"discussion"` into the `"title"` field.
* **Mitigation:** Synthesize training variations pairing `"{title} discussion"`, `"{title} meeting"`, and `"{title} session"`.

### Failure Case 3: Default Slot Hallucination on Missing Information
* **Sample ID:** `test_0014`
* **Input Query:** `"Hold a meeting with Kenji and Tariq at for quick regarding Customer advisory council."`
* **Expected JSON:**
  ```json
  {"intent": "schedule_meeting", "title": "Customer advisory council", "participants": ["Kenji", "Tariq"], "date": null, "time": null, "duration_minutes": null}
  ```
* **Actual Output:**
  ```json
  {"intent": "schedule_meeting", "title": "Customer advisory council", "participants": ["Kenji", "Tariq"], "date": null, "time": "12:00", "duration_minutes": 30}
  ```
* **Category:** Default Slot Hallucination
* **Root Cause:** The phrase `"for quick"` lacked numeric duration and time, but the model defaulted to conversational prior values (`time: "12:00"`, `duration_minutes: 30`).
* **Mitigation:** Add negative training pairs explicitly demonstrating that non-numeric qualifiers (`"quick"`, `"soon"`) must map to `null`.

---

## 10. 7B Parameter Model GPU Memory Calculation

*(Summary of mathematical analysis from [`reports/memory_calculation.md`](reports/memory_calculation.md))*

| Fine-Tuning Paradigm | Base Model Precision | Trainable Params | Static State Memory | Activation Memory ($s=2048, b=2$) | Practical Peak GPU Memory | Hardware Requirement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full Fine-Tuning** | 16-bit (bf16/fp16) | 7.0B (100%) | **112.0 GB / 104.3 GiB** | 3.5 GB / 3.3 GiB | **~118.0 GB / ~110.0 GiB** | $2\times$ A100 80GB (FSDP/ZeRO-3) |
| **LoRA (Frozen Base)** | 16-bit (bf16/fp16) | ~30M (~0.43%) | **14.5 GB / 13.5 GiB** | 3.8 GB / 3.5 GiB | **~20.3 GB / ~18.9 GiB** | $1\times$ RTX 3090/4090 24GB or A10G |
| **QLoRA (4-bit NF4)** | 4-bit NormalFloat4 | ~30M (~0.43%) | **4.7 GB / 4.4 GiB** | 3.8 GB / 3.5 GiB | **~10.0 GB / ~9.3 GiB** | $1\times$ RTX 3060/4060 12GB or Colab T4 |

* **Formula for Full Fine-Tuning Static State:**
  $$\text{Memory} = 2 \text{ bytes (weight)} + 2 \text{ bytes (grad)} + 12 \text{ bytes (AdamW)} = 16 \text{ bytes/param} \times 7\times 10^9 = \mathbf{112 \text{ GB}}$$
* **Formula for QLoRA Static State:**
  $$\text{Memory} = 0.5 \text{ bytes (4-bit base)} + 0.13 \text{ bytes (DQ)} + 16 \text{ bytes/param} \times 30\times 10^6 \text{ (adapter)} = \mathbf{4.71 \text{ GB}}$$

---

## 11. How to Reproduce the Experiment

### 11.1 Local Execution
```bash
# Clone and navigate to Task 1 directory
git clone https://github.com/aditya-hariom/tndrr-ai-internship-assignment.git
cd tndrr-ai-internship-assignment/task-1

# Install dependencies
pip install -r requirements.txt

# Step 1: Generate dataset & run leakage audit
python scripts/generate_dataset.py
python scripts/validate_dataset.py

# Step 2: Evaluate untuned baseline model
python src/evaluate.py --test_data data/test.jsonl

# Step 3: Run LoRA fine-tuning
python src/train.py

# Step 4: Evaluate fine-tuned model & generate comparison
python src/evaluate.py --adapter_path models/qwen2.5-0.5b-lora/best_adapter
python scripts/compare_models.py
```

### 11.2 Google Colab / Kaggle Execution
Open and execute [`notebooks/finetuning_experiment.ipynb`](notebooks/finetuning_experiment.ipynb) in a single click on a free Colab T4 GPU.

---

## 12. Hardware & Software Specifications

* **Operating System:** Windows 11
* **GPU:** NVIDIA GeForce RTX 2050 Laptop GPU (4,096 MiB VRAM, 55W TGP, Compute Capability 8.6 Ampere)
* **CUDA Driver:** Driver 552.27, CUDA 12.4
* **Python Environment:** Python 3.13.5 with `torch-2.6.0+cu124`
* **Core Libraries:** `transformers 5.23.0`, `peft 0.21.3`, `accelerate 1.16.0`

---

## 13. Limitations & Future Roadmap

1. **Schema-Constrained Decoding:** While LoRA achieved 99.38% JSON validity, a small probability of malformed JSON remains. Implementing finite-state machine (FSM) grammar sampling (via Outlines or LM-Format-Enforcer) guarantees 100% syntactic validity.
2. **Context Window Expansion:** Current evaluation uses sequence length 256. Adapting to long conversational histories (multi-turn dialogues with conversational context) would enhance real-world assistant usage.
3. **Direct Preference Optimization (DPO):** Applying DPO on top of SFT with negative pairs for ambiguous utterances would further eliminate out-of-schema intent hallucinations.
