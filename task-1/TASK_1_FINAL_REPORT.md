# TASK 1 – Parameter-Efficient Fine-Tuning (LoRA) for Structured JSON Parsing
## Complete Academic Report & Experimental Investigation

**Course:** AI & Large Language Models Internship  
**Experiment Date:** November 19, 2026  
**Hardware Platform:** NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM) + AMD Ryzen 7 5800HS + 16 GB System RAM  
**Software Platform:** Windows 11 (64-bit), PyTorch 2.6.0+cu124, Transformers 4.49.0 / 5.23.0, PEFT 0.21.3  

---

## 1. Project Title & Abstract

### 1.1 Project Title
**Reproducible Parameter-Efficient Fine-Tuning (PEFT): Transforming Small Language Models into Deterministic JSON Function-Calling Engines via LoRA**

### 1.2 Abstract
Deploying Large Language Models (LLMs) in automated agentic workflows, API routing, and mobile edge microservices demands deterministic conversion of unstructured natural language utterances into strictly typed, schema-compliant JSON function calls. While modern foundation models possess strong conversational capabilities, untuned lightweight Small Language Models (SLMs $\le 0.5\text{B}$) frequently fail on structured extraction tasks—hallucinating conversational conversational preambles, fabricating unsupported schema keys, hallucinating default values for missing attributes, and omitting required fields.

This project implements a production-grade, fully reproducible Parameter-Efficient Fine-Tuning (PEFT) pipeline using **Low-Rank Adaptation (LoRA)** on **`Qwen/Qwen2.5-0.5B-Instruct`** (494M parameters). We construct a leak-free synthetic dataset of 1,600 examples partitioned into 1,280 training, 160 validation, and 160 held-out test samples with 0 query collisions and disjoint entity partitions. By training only **2,162,688 parameters (0.4359% of total weights / 8.67 MB)** with prompt-loss masking, LoRA elevates Intent Classification Accuracy from **0.00% to 98.75%**, Exact Match Accuracy from **0.00% to 64.38%**, Field Micro F1 from **5.97% to 86.75%**, and Raw JSON Validity from **78.12% to 99.38%**.

Furthermore, this report presents an exhaustive failure diagnosis on real test errors, provides step-by-step reproduction instructions for Google Colab / Kaggle, and develops a rigorous first-principles mathematical GPU memory derivation comparing Full Fine-Tuning (118 GB), LoRA (20.3 GB), and QLoRA (10.0 GB) on a 7-Billion parameter language model.

---

## 2. Problem Statement & Objectives

### 2.1 The Challenge of Structured Generation
In autonomous agent architectures, language models function as policy deciders that route requests to external tools, APIs, and databases. General-purpose instruction-tuned models suffer from systematic failure modes when required to emit pure JSON:
1. **Markdown Wrapping & Conversational Noise:** Emitting conversational preambles (`"Certainly! Here is your JSON:"`) or markdown fences (````json ... ````) that crash strict downstream parsers.
2. **Schema Drift & Key Hallucination:** Inventing plausible-sounding keys (`"action"`, `"target_time"`, `"meeting_name"`) rather than strictly following the API specification (`"intent"`, `"time"`, `"title"`).
3. **Default Value Fabrication:** Populating unmentioned optional fields with arbitrary defaults (e.g., assuming a meeting lasts 30 minutes when no duration was requested) instead of explicitly outputting `null`.
4. **Data Type Mismatches:** Emitting human string representations (e.g. `"15 minutes"`, `"$20"`) instead of integer seconds or float amounts (`900`, `20.0`).

### 2.2 Project Objectives
1. **Model Selection:** Select a lightweight, open-weight base SLM ($\le 0.5\text{B}$ parameters) capable of running and training on consumer hardware.
2. **Deterministic Schema Formulation:** Define a rigorous 4-intent taxonomy plus an `unknown` fallback schema enforcing explicit `null` emissions for absent slots.
3. **Leak-Free Dataset Synthesis:** Construct a stratified dataset of $\ge 1,000$ examples with automated zero-leakage verification (0 exact collisions, disjoint entity sets, cross-split 4-gram Jaccard overlap $< 0.01$).
4. **PEFT Implementation:** Train a low-rank adapter ($r=16, \alpha=32$) targeting all attention projections with strict prompt-loss masking.
5. **Objective Evaluation:** Evaluate base vs fine-tuned models on a held-out test set ($\ge 50$ samples, 160 used) under greedy deterministic decoding across JSON validity, Intent Accuracy, Canonical Exact Match, and Field Micro F1.
6. **Failure Analysis:** Dissect real failure modes to identify systemic root causes and production mitigations.
7. **Theoretical 7B Memory Analysis:** Derive exact static state, activation, and peak VRAM requirements for 7B model fine-tuning across Full FT, LoRA, and QLoRA.

---

## 3. Base Model Specification

The project leverages **`Qwen/Qwen2.5-0.5B-Instruct`**, released by Alibaba Cloud:

| Property | Value | Architectural Significance |
| :--- | :--- | :--- |
| **Model Architecture** | `Qwen2ForCausalLM` | Modern decoder-only transformer with SwiGLU activations and RMSNorm. |
| **Total Parameter Count** | 494,032,768 parameters | Compact footprint suited for resource-constrained edge devices and 4 GB VRAM GPUs. |
| **Hidden Dimension ($d_{\text{model}}$)** | 896 | Compact vector space facilitating rapid gradient propagation. |
| **Number of Layers** | 24 transformer blocks | Sufficient depth for hierarchical syntactic and semantic understanding. |
| **Attention Heads / KV Heads** | 14 Query / 2 KV Heads | Grouped Query Attention (GQA) dramatically reduces KV cache footprint during generation. |
| **Positional Embeddings** | RoPE (Rotary Position Embeddings) | Relative positional awareness preserving entity token order. |
| **Vocabulary Size** | 151,936 tokens | Large BPE vocabulary enabling concise tokenization of numbers and punctuation. |
| **Prompt Format** | ChatML (`<\|im_start\|>...<\|im_end\|>`) | Industry-standard role delineators separating system, user, and assistant turns. |

---

## 4. Predefined JSON Schema & Intent Taxonomy

The model is trained to parse natural language user requests into one of five structured schemas:

```json
{
  "create_reminder": {
    "required": ["intent", "task"],
    "optional": ["date", "time", "priority"],
    "priority_values": ["low", "medium", "high", null]
  },
  "schedule_meeting": {
    "required": ["intent", "title", "participants"],
    "optional": ["date", "time", "duration_minutes"],
    "participants_format": "array of strings"
  },
  "set_timer": {
    "required": ["intent", "duration_seconds"],
    "optional": ["label"],
    "duration_seconds_format": "integer"
  },
  "log_expense": {
    "required": ["intent", "amount", "currency"],
    "optional": ["category", "merchant"],
    "amount_format": "float",
    "currency_format": "standard currency code (USD, EUR, INR, etc.)"
  },
  "unknown": {
    "required": ["intent", "error", "raw_query"],
    "intent_value": "unknown"
  }
}
```

### The Explicit Null Rule
Unlike permissive REST endpoints that omit undefined fields, our system enforces:
$$\forall s \in \text{OptionalSlots}, \quad s \notin \text{Utterance} \implies \text{output}[s] = \text{null}$$
This prevents downstream microservices from having to distinguish between missing keys and undefined variables, eliminating entire classes of unhandled KeyError exceptions.

---

## 5. Dataset Engineering & Zero-Leakage Architecture

### 5.1 Dataset Stratification
A total of **1,600 examples** were generated with a fixed random seed (`seed=42`) using `scripts/generate_dataset.py`:

| Dataset Split | File Path | Sample Count | Percentage | Minimum Required | Conformance |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Training Set** | `data/train.jsonl` | 1,280 | 80.0% | $\ge 1,000$ | Exceeded |
| **Validation Set** | `data/validation.jsonl` | 160 | 10.0% | $\ge 50$ | Exceeded |
| **Held-Out Test Set** | `data/test.jsonl` | 160 | 10.0% | $\ge 50$ | Exceeded |
| **Total** | | **1,600** | **100.0%** | | |

Intent distribution is perfectly balanced across all three splits:
* `create_reminder`: 22.5%
* `schedule_meeting`: 22.5%
* `set_timer`: 22.5%
* `log_expense`: 22.5%
* `unknown` (fallback/out-of-domain): 10.0%

### 5.2 Zero-Leakage Verification Protocol
To guarantee genuine generalisation rather than pattern memorisation, `scripts/validate_dataset.py` enforces four orthogonal guarantees:

1. **Global Deduplication Audit:** A cryptographic hash set verified that **0 verbatim queries** overlap across `train`, `validation`, and `test` splits.
2. **Disjoint Entity Partitions:**
   * **Participants:** Names in `train` (e.g., *Aarav, Liam, Sophia, Rohan*) are completely disjoint from `validation` (*Priya, Noah, Maya*) and `test` (*Kenji, Amara, Tariq, Kavita*).
   * **Merchants:** Companies in `train` (*Starbucks, Amazon, Uber*) do not appear in `test` (*Trader Joe's, Delta Airlines, Coursera*).
   * **Meeting Titles:** Project topics are partitioned across splits.
3. **Disjoint Template Families:**
   * `train` emphasizes direct command structures (*"Create a reminder to...", "Schedule a meeting with..."*).
   * `test` utilizes polite requests, elliptical phrases, and colloquial variations (*"Could you please ping me to...", "Block 45m with..."*).
4. **Lexical Isolation Metric:**
   $$\text{Jaccard}_{4\text{-gram}}(A, B) = \frac{|N_4(A) \cap N_4(B)|}{|N_4(A) \cup N_4(B)|}$$
   Measured results:
   * $\text{Jaccard}_{4\text{-gram}}(\text{train}, \text{validation}) = \mathbf{0.0004}$
   * $\text{Jaccard}_{4\text{-gram}}(\text{train}, \text{test}) = \mathbf{0.0089}$

---

## 6. Training Methodology & LoRA Configuration

### 6.1 Low-Rank Adaptation (LoRA) Mathematical Principles
Given a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, full fine-tuning modifies weights via $\Delta W \in \mathbb{R}^{d \times k}$. LoRA decomposes the update into the product of two low-rank matrices:
$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \cdot A)$$
where $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$.
* Matrix $A$ is initialized from a Gaussian distribution $\mathcal{N}(0, \sigma^2)$.
* Matrix $B$ is initialized to zero, ensuring $\Delta W = 0$ at step 0 (no disruption of pre-trained capabilities).
* $\alpha$ is a constant scaling hyperparameter; the ratio $\frac{\alpha}{r}$ stabilizes updates when varying $r$.

### 6.2 LoRA Hyperparameters

| Hyperparameter | Value | Technical Justification |
| :--- | :--- | :--- |
| **LoRA Rank ($r$)** | `16` | Provides sufficient bottleneck capacity for multi-slot JSON key mapping. |
| **LoRA Alpha ($\alpha$)** | `32` | Standard scaling ratio $\frac{\alpha}{r} = 2.0$, balancing base activations and adapter updates. |
| **LoRA Dropout** | `0.05` | Injects activation noise during training to prevent overfitting to specific slot tokens. |
| **Target Modules** | `q_proj, k_proj, v_proj, o_proj` | Adapts all 4 multi-head attention projection layers, enabling cross-token semantic routing. |
| **Trainable Parameters** | **2,162,688 (0.4359%)** | Over 99.56% of the base model remains completely frozen. |
| **Adapter Storage Size** | **8.67 MB** (`adapter_model.safetensors`) | Compact enough to be committed directly to Git, distributed via CDN, or deployed to mobile apps. |

### 6.3 Prompt Loss Masking
Standard autoregressive language modeling computes cross-entropy over all sequence tokens. In our implementation (`src/dataset.py`):
$$\mathcal{L} = -\frac{1}{|\mathcal{T}_{\text{target}}|} \sum_{t \in \mathcal{T}_{\text{target}}} \log P(x_t \mid x_{<t})$$
Tokens belonging to the system prompt and user input are masked with `label = -100` (`IGNORE_INDEX`). Cross-entropy is evaluated **exclusively on the target assistant JSON tokens**, preventing the adapter from wasting rank capacity learning to generate natural language instructions.

### 6.4 Training Execution Profile
* **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay $= 0.01$)
* **Learning Rate:** $2.0 \times 10^{-4}$ with linear warmup (5% steps) and Cosine Annealing decay to 0
* **Batch Configuration:** Micro-batch size 2 with 4 gradient accumulation steps $\implies$ **Effective Batch Size = 8**
* **Epochs:** 3 full epochs (480 total optimizer steps)
* **Training Time:** 802.79 seconds (~13.4 minutes) on local NVIDIA RTX 2050 Laptop GPU (55W TGP)

#### Training Loss Trajectory:
* Initial Validation Loss: `1.5307`
* **Epoch 1:** Train Loss `0.2317` | Val Loss `0.0654` | LR `1.59e-4`
* **Epoch 2:** Train Loss `0.0533` | **Val Loss 0.0550 (Best)** | LR `5.50e-5`
* **Epoch 3:** Train Loss `0.0422` | Val Loss `0.0571` | LR `0.00`
* *Checkpoint Selection:* Best adapter saved from Epoch 2 (`val_loss = 0.0550`).

---

## 7. Evaluation Methodology & Mathematical Metrics

Evaluation was conducted on all **160 held-out test samples** under strictly deterministic generation conditions:
$$\text{Sampling Strategy: Greedy Decoding } (\text{do\_sample} = \text{False}, \text{temperature} = 0.0, \text{max\_new\_tokens} = 80)$$

Four objective mathematical metrics were computed:

1. **Raw JSON Validity Rate ($\text{JVR}_{\text{raw}}$):**
   $$\text{JVR}_{\text{raw}} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{json.loads}(\hat{y}_i) \text{ succeeds without error})$$
2. **Intent Classification Accuracy ($\text{ICA}$):**
   $$\text{ICA} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i[\text{"intent"}] = y_i^*[\text{"intent"}])$$
3. **Canonical Exact Match Accuracy ($\text{EMA}$):**
   $$\text{EMA} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\text{Canonical}(\hat{y}_i) \equiv \text{Canonical}(y_i^*))$$
   where $\text{Canonical}(\cdot)$ sorts all dictionary keys recursively, normalizes numerical formats ($15.0 \equiv 15$), and strips whitespace.
4. **Field Micro Precision, Recall, and F1 Score:**
   Computed over all key-value slots excluding `"intent"`:
   $$\text{Precision}_{\text{micro}} = \frac{\sum_{k} \text{TP}_k}{\sum_k (\text{TP}_k + \text{FP}_k)}, \quad \text{Recall}_{\text{micro}} = \frac{\sum_k \text{TP}_k}{\sum_k (\text{TP}_k + \text{FN}_k)}$$
   $$\text{F1}_{\text{micro}} = \frac{2 \cdot \text{Precision}_{\text{micro}} \cdot \text{Recall}_{\text{micro}}}{\text{Precision}_{\text{micro}} + \text{Recall}_{\text{micro}}}$$

---

## 8. Experimental Results: Base vs. Fine-Tuned

The table below reports actual experimental measurements obtained on the 160 held-out test examples:

| Evaluation Metric | Untuned Base Model | LoRA Fine-Tuned Model | Absolute Gain ($\Delta$) | Relative Gain |
| :--- | :---: | :---: | :---: | :---: |
| **Raw JSON Validity Rate** | 78.12% | **99.38%** | **+21.26%** | **+27.2%** |
| **Extracted JSON Validity** | 85.62% | **99.38%** | **+13.76%** | **+16.1%** |
| **Intent Classification Accuracy** | 0.00% | **98.75%** | **+98.75%** | *Baseline 0%* |
| **Exact Match (EM) Accuracy** | 0.00% | **64.38%** | **+64.38%** | *Baseline 0%* |
| **Field Micro Precision** | 8.28% | **86.98%** | **+78.70%** | **+950.5%** |
| **Field Micro Recall** | 4.66% | **86.53%** | **+81.87%** | **+1,756.9%** |
| **Field Micro F1 Score** | **5.97%** | **86.75%** | **+80.78%** | **+1,353.1%** |
| **Field Micro Accuracy** | 3.08% | **76.61%** | **+73.53%** | **+2,387.3%** |

```
=============================================================================================
                              EXPERIMENTAL PERFORMANCE BENCHMARK
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

### Key Analytical Takeaways:
1. **Why Base Model Scored 0.0% Intent Accuracy:** The base model generated JSON containing arbitrary conversational intent names (e.g. `"reminder_request"`, `"calendar_invite"`, `"alarm"`) or embedded the intent inside descriptive English rather than utilizing our exact canonical strings (`"create_reminder"`, `"schedule_meeting"`, `"set_timer"`, `"log_expense"`).
2. **Deterministic Output Formatting:** LoRA eliminated conversational output preambles and markdown fencing entirely, lifting raw JSON validity from 78.12% to 99.38%.
3. **High Slot Extraction Generalisation:** Achieving 86.75% Field Micro F1 on completely unseen entity names and phrasing structures proves that LoRA learned the underlying semantic slot syntax rather than memorizing training tokens.

---

## 9. Qualitative Failure Analysis

To understand model boundaries, we analyze three genuine errors extracted directly from `results/finetuned_predictions.jsonl`:

### Failure Case 1: Out-of-Schema Intent Hallucination
* **Sample ID:** `test_0018`
* **Input Query:** `"Track twenty dollars."`
* **Ground Truth JSON:**
  ```json
  {
    "intent": "unknown",
    "error": "insufficient_information",
    "raw_query": "Track twenty dollars."
  }
  ```
* **Model Output:**
  ```json
  {
    "intent": "set_goal",
    "amount": 20.0,
    "category": null,
    "label": "Track"
  }
  ```
* **Failure Category:** Ambiguous Request / Out-of-Schema Intent Hallucination
* **Root Cause:** The query mentions a dollar amount without indicating whether it represents an expense, transfer, or savings goal. Rather than falling back to the schema-defined `"unknown"` intent, the model attempted to be overly cooperative, hallucinating an intent `"set_goal"` not defined anywhere in the training taxonomy.
* **Production Mitigation:** Incorporate negative contrastive training pairs and apply schema-constrained decoding (e.g., Outlines / Guidance / LM-Format-Enforcer) that restricts the vocabulary logits to the exact 5 valid intent tokens at the `"intent"` key position.

---

### Failure Case 2: Entity Span Boundary Bleed (Over-Extraction)
* **Sample ID:** `test_0006`
* **Input Query:** `"Set up a Customer advisory council discussion with Kavita and Amara on on Jan 15th at 10:00 AM for 90-minute."`
* **Ground Truth JSON:**
  ```json
  {
    "intent": "schedule_meeting",
    "title": "Customer advisory council",
    "participants": ["Kavita", "Amara"],
    "date": "on Jan 15th",
    "time": "10:00",
    "duration_minutes": 90
  }
  ```
* **Model Output:**
  ```json
  {
    "intent": "schedule_meeting",
    "title": "Customer advisory council discussion",
    "participants": ["Kavita", "Amara"],
    "date": "on Jan 15th",
    "time": "10:00",
    "duration_minutes": 90
  }
  ```
* **Failure Category:** Entity Span Boundary Delineation
* **Root Cause:** The model absorbed the structural meeting descriptor `"discussion"` into the `"title"` slot (`"Customer advisory council discussion"` instead of `"Customer advisory council"`).
* **Production Mitigation:** Synthesize training variations pairing interchangeable structural nouns (`"{topic} discussion"`, `"{topic} meeting"`, `"{topic} sync"`) where the label explicitly strips the trailing meeting noun.

---

### Failure Case 3: Default Slot Hallucination on Missing Quantitative Data
* **Sample ID:** `test_0014`
* **Input Query:** `"Hold a meeting with Kenji and Tariq at for quick regarding Customer advisory council."`
* **Ground Truth JSON:**
  ```json
  {
    "intent": "schedule_meeting",
    "title": "Customer advisory council",
    "participants": ["Kenji", "Tariq"],
    "date": null,
    "time": null,
    "duration_minutes": null
  }
  ```
* **Model Output:**
  ```json
  {
    "intent": "schedule_meeting",
    "title": "Customer advisory council",
    "participants": ["Kenji", "Tariq"],
    "date": null,
    "time": "12:00",
    "duration_minutes": 30
  }
  ```
* **Failure Category:** Default Slot Hallucination
* **Root Cause:** The query contained broken colloquial timing (`"at for quick"`). Instead of adhering to the strict rule that missing slots must evaluate to `null`, the model hallucinated conversational default priors (`"time": "12:00"`, `"duration_minutes": 30`).
* **Production Mitigation:** Augment training data with ungrammatical, fragmented, and hurried user queries, reinforcing that conversational temporal qualifiers like `"quick"`, `"soon"`, and `"brief"` must map to `duration_minutes: null` unless a concrete number is supplied.

---

## 10. Mathematical GPU Memory Estimation for a 7B Parameter Model

*(Detailed theoretical derivation comparing Full Fine-Tuning, LoRA, and QLoRA)*

### 10.1 Standard Production Training Assumptions
* **Model Size:** $N = 7.0 \times 10^9$ parameters (e.g., Llama-2-7B, Qwen-7B)
* **Architecture:** $L = 32$ layers, $h = 4096$ hidden dimension, $a = 32$ attention heads
* **Sequence Length:** $s = 2048$ tokens
* **Per-Device Micro-Batch Size:** $b = 2$
* **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$)
* **Activation Optimization:** Gradient Checkpointing + FlashAttention-2 enabled

---

### 10.2 Parameter States (Static Memory Equations)
For any trainable parameter $\theta$:
1. **Model Weight:** 2 bytes (`bfloat16`/`float16`)
2. **Weight Gradient:** 2 bytes (`bfloat16`/`float16`)
3. **AdamW Optimizer States:** 12 bytes
   * Master copy of weight in fp32: 4 bytes
   * First momentum vector ($m_t$ in fp32): 4 bytes
   * Second uncentered variance ($v_t$ in fp32): 4 bytes
$$\text{Static State per Trainable Parameter} = 2 + 2 + 12 = \mathbf{16 \text{ bytes/parameter}}$$

---

### 10.3 Comparative Paradigm Breakdown

| Metric / Component | A. Full Fine-Tuning (16-bit) | B. LoRA (16-bit Frozen Base) | C. QLoRA (4-bit NF4 Base) |
| :--- | :---: | :---: | :---: |
| **Base Model Precision** | 16-bit (`bf16`/`fp16`) | 16-bit (`bf16`/`fp16`) | 4-bit NormalFloat4 (NF4) |
| **Trainable Parameters** | 7.0B (100%) | ~30M (~0.43%) | ~30M (~0.43%) |
| **Base Model Weights** | $7.0\text{B} \times 2\text{B} = \mathbf{14.00 \text{ GB}}$ | $7.0\text{B} \times 2\text{B} = \mathbf{14.00 \text{ GB}}$ | $7.0\text{B} \times 0.5\text{B} = \mathbf{3.50 \text{ GB}}$ |
| **Double Quantization (DQ) Metadata** | $0 \text{ GB}$ | $0 \text{ GB}$ | $7.0\text{B} \times 0.13\text{B} \approx \mathbf{0.91 \text{ GB}}$ |
| **Adapter Weights** | $0 \text{ GB}$ | $30\text{M} \times 2\text{B} = \mathbf{0.06 \text{ GB}}$ | $30\text{M} \times 2\text{B} = \mathbf{0.06 \text{ GB}}$ |
| **Gradients** | $7.0\text{B} \times 2\text{B} = \mathbf{14.00 \text{ GB}}$ | $30\text{M} \times 2\text{B} = \mathbf{0.06 \text{ GB}}$ | $30\text{M} \times 2\text{B} = \mathbf{0.06 \text{ GB}}$ |
| **AdamW Optimizer States** | $7.0\text{B} \times 12\text{B} = \mathbf{84.00 \text{ GB}}$ | $30\text{M} \times 12\text{B} = \mathbf{0.36 \text{ GB}}$ | $30\text{M} \times 6\text{B} = \mathbf{0.18 \text{ GB}}$ *(Paged)* |
| **Total Static State Memory** | **112.00 GB (104.31 GiB)** | **14.48 GB (13.49 GiB)** | **4.71 GB (4.39 GiB)** |
| **Activation Memory ($b=2, s=2048$)** | 3.50 GB (3.26 GiB) | 3.80 GB (3.54 GiB) | 3.80 GB (3.54 GiB) |
| **CUDA Workspace & Context** | 2.50 GB (2.33 GiB) | 2.00 GB (1.86 GiB) | 1.50 GB (1.40 GiB) |
| **Practical Peak GPU Memory** | **~118.0 GB (~110.0 GiB)** | **~20.3 GB (~18.9 GiB)** | **~10.0 GB (~9.3 GiB)** |
| **Hardware Viability** | $2\times$ NVIDIA A100 80GB (FSDP/ZeRO-3) | $1\times$ NVIDIA RTX 3090/4090 24GB or A10G | $1\times$ RTX 3060/4060 12GB or Google Colab T4 |

### 10.4 Activation Memory Formula & Derivation
With **Gradient Checkpointing**, intermediate activations within transformer blocks are recomputed during backward pass rather than stored:
$$\text{Memory}_{\text{act, checkpointed}} \approx 2 \cdot b \cdot s \cdot h \cdot \sqrt{L} \text{ bytes}$$
For $b=2, s=2048, h=4096, L=32$:
$$\text{Memory}_{\text{act}} \approx 2 \cdot 2 \cdot 2048 \cdot 4096 \cdot \sqrt{32} \cdot 2 \text{ bytes} \approx 3.79 \times 10^9 \text{ bytes} \approx \mathbf{3.53 \text{ GiB}} \; (3.79 \text{ GB})$$
Without gradient checkpointing, storing activations across all 32 layers requires $> 32 \text{ GiB}$, which would instantly cause an Out-Of-Memory (OOM) fault on consumer hardware.

---

## 11. Reproduction Guide

### 11.1 Local CUDA / CPU Execution
```bash
# Clone repository and enter Task 1 directory
git clone https://github.com/aditya-hariom/tndrr-ai-internship-assignment.git
cd tndrr-ai-internship-assignment/task-1

# Install requirements
pip install -r requirements.txt

# Step 1: Synthesize dataset and run zero-leakage audit
python scripts/generate_dataset.py
python scripts/validate_dataset.py

# Step 2: Evaluate untuned base model
python src/evaluate.py --test_data data/test.jsonl

# Step 3: Run LoRA fine-tuning
python src/train.py

# Step 4: Evaluate fine-tuned LoRA model & compare
python src/evaluate.py --adapter_path models/qwen2.5-0.5b-lora/best_adapter
python scripts/compare_models.py
```

### 11.2 Interactive Jupyter / Google Colab Execution
Open [`notebooks/finetuning_experiment.ipynb`](notebooks/finetuning_experiment.ipynb) in Google Colab or Kaggle. The notebook runs turn-key end-to-end on a free T4 GPU in under 15 minutes.

---

## 12. Viva Voce Defense & Technical Q&A Cheat Sheet

| Question | Defensible Technical Answer |
| :--- | :--- |
| **Q1: Why choose LoRA over full fine-tuning?** | Full fine-tuning updates all 494M weights and requires 16 bytes/param for AdamW states, demanding significant GPU memory and causing catastrophic forgetting. LoRA freezes 99.56% of weights, adding low-rank matrices ($r=16$) to attention projections ($0.4359\%$ trainable params), saving 85% memory while matching full FT performance. |
| **Q2: Why did the base model achieve 0.0% Intent Accuracy?** | The base model did not have conversational training on our exact intent strings. While it attempted structured output, it emitted arbitrary non-standard keys (`"reminder_task"`, `"alarm"`) or embedded intents in conversational prose, failing strict canonical exact match. |
| **Q3: What does Prompt Loss Masking achieve?** | Standard language modeling minimizes loss across all sequence tokens. In structured generation, the user query and system prompt are fixed inputs. Setting their labels to `-100` forces cross-entropy optimization solely over the JSON target tokens, focusing model capacity strictly on output synthesis. |
| **Q4: How does QLoRA achieve 4-bit storage without destroying precision?** | QLoRA uses NormalFloat4 (NF4), an information-theoretically optimal quantile quantization for zero-mean normal distributions. Weights are stored in 4-bit, dequantized on-the-fly to 16-bit for forward/backward computation, and gradients are accumulated in 16-bit LoRA adapters. |
| **Q5: How did you ensure your test evaluation was not leaking?** | We enforced: (1) zero query string collisions via global set deduplication; (2) disjoint entity partitions for names, merchants, and dates; (3) disjoint phrasing templates; and (4) verified cross-split 4-gram Jaccard overlap was $< 0.009$. |

---

## 13. Conclusion & Limitations

### 13.1 Conclusion
This project successfully demonstrated that fine-tuning a small 494M parameter language model with LoRA transforms it into a highly deterministic JSON function-calling agent, improving Field Micro F1 by **+1,353%** and Intent Accuracy from **0.0% to 98.75%** with an 8.67 MB adapter footprint.

### 13.2 Limitations & Production Roadmap
1. **Grammar-Constrained Decoding:** In 0.62% of test cases, minor JSON syntax anomalies occurred. Integrating Outlines or LM-Format-Enforcer guarantees 100% syntactic validity via finite-state machine (FSM) logit masking.
2. **Direct Preference Optimization (DPO):** To eliminate the 3 failure modes identified in Section 9, future work will train a DPO reward head using contrastive pairs that map ambiguous utterances strictly to `unknown`.
