# Mathematical GPU Memory Estimation for Fine-Tuning a 7B Language Model

**Author:** Senior ML Engineer  
**Scope:** Deep-dive memory analysis comparing **Full Fine-Tuning**, **LoRA**, and **QLoRA** on a 7-Billion Parameter Model.

---

## 1. Executive Summary & Quick Reference

The table below summarizes the GPU memory requirements for fine-tuning a 7-billion parameter language model ($N = 7.0 \times 10^9$) under standard production training configurations:
* **Sequence Length:** $s = 2048$
* **Per-Device Micro-Batch Size:** $b = 2$
* **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$)
* **Activation Optimization:** Gradient Checkpointing enabled
* **Attention Kernel:** FlashAttention-2 enabled

| Fine-Tuning Paradigm | Base Model Precision | Trainable Params | Static State Memory | Activation Memory ($s=2048, b=2$) | CUDA Context & Workspace | Total Practical Peak GPU Memory | Hardware Requirement |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Full Fine-Tuning** | 16-bit (bf16/fp16) | 7.0B (100%) | **112.0 GB / 104.3 GiB** | 3.5 GB / 3.3 GiB | 2.5 GB / 2.3 GiB | **~118.0 GB / ~110.0 GiB** | $2\times$ A100 80GB (FSDP/ZeRO-3) |
| **B. LoRA (Frozen Base)** | 16-bit (bf16/fp16) | ~30M (~0.43%) | **14.5 GB / 13.5 GiB** | 3.8 GB / 3.5 GiB | 2.0 GB / 1.9 GiB | **~20.3 GB / ~18.9 GiB** | $1\times$ RTX 3090/4090 24GB or A10G |
| **C. QLoRA (4-bit NF4)** | 4-bit NormalFloat4 | ~30M (~0.43%) | **4.7 GB / 4.4 GiB** | 3.8 GB / 3.5 GiB | 1.5 GB / 1.4 GiB | **~10.0 GB / ~9.3 GiB** | $1\times$ RTX 3060/4060 12GB or Colab T4 16GB |

> [!NOTE]
> **Unit Clarification:**  
> Decimal: $1 \text{ GB} = 10^9 \text{ bytes} = 1,000,000,000 \text{ bytes}$.  
> Binary: $1 \text{ GiB} = 2^{30} \text{ bytes} = 1,073,741,824 \text{ bytes}$.  
> Conversion factor: $\text{GiB} = \text{GB} \div 1.073741824$. GPU manufacturers market cards in GB, while `nvidia-smi` and PyTorch report in GiB / MiB.

---

## 2. Anatomy of GPU Memory in LLM Fine-Tuning

Total peak GPU memory during training is governed by:

$$\text{Memory}_{\text{total}} = \text{Memory}_{\text{model}} + \text{Memory}_{\text{gradients}} + \text{Memory}_{\text{optimizer}} + \text{Memory}_{\text{activations}} + \text{Memory}_{\text{overhead}}$$

### 2.1 Parameter States (Static Memory)
For every trainable parameter $\theta_i$:
1. **Model Weight:** Stored in 16-bit (`bfloat16`/`float16`) = 2 bytes/param.
2. **Gradient:** Stored in 16-bit or 32-bit = 2 or 4 bytes/param.
3. **AdamW Optimizer States:**
   * Master copy of weight in fp32: 4 bytes/param (prevents underflow during tiny gradient steps).
   * First momentum ($m_t$ in fp32): 4 bytes/param.
   * Second uncentered variance ($v_t$ in fp32): 4 bytes/param.
   * **Total AdamW state = 12 bytes per trainable parameter.**

$$\text{Static State per Trainable Param} = 2 \text{ (weight)} + 2 \text{ (grad)} + 12 \text{ (AdamW)} = 16 \text{ bytes/param}$$

*(If fp32 gradient accumulation is enforced: $2 + 4 + 12 = 18 \text{ bytes/param}$).*

### 2.2 Activation Memory (Dynamic Memory)
Activations are the intermediate forward-pass tensor values stored to compute gradients during the backward pass (layer inputs, attention matrices, MLP projections).

* **Without Gradient Checkpointing:**
  Activations must be stored for all $L$ layers:
  $$\text{Memory}_{\text{act, standard}} \approx b \cdot s \cdot h \cdot L \cdot (34 + 5 \cdot \frac{a \cdot s}{h}) \text{ bytes}$$
  For a 7B model ($L=32, h=4096, a=32$ heads), with $b=2, s=2048$, this exceeds **32 GiB**!
* **With Gradient Checkpointing (Activation Recomputation):**
  Only activations at layer boundaries are stored; internal activations are recomputed during backward pass:
  $$\text{Memory}_{\text{act, checkpointed}} \approx 2 \cdot b \cdot s \cdot h \cdot \sqrt{L} \text{ bytes}$$
  For $b=2, s=2048, h=4096, L=32$:
  $$\approx 2 \cdot 2 \cdot 2048 \cdot 4096 \cdot \sqrt{32} \cdot 2 \text{ bytes} \approx 3.79 \times 10^9 \text{ bytes} \approx 3.53 \text{ GiB}$$

---

## 3. Detailed Mathematical Working for 7B Model

We assume a standard 7B architecture (e.g., Llama-2-7B / Qwen-7B):
* Total base parameters: $N_{\text{base}} = 7.0 \times 10^9$
* Layers $L = 32$, Hidden dimension $h = 4096$, Intermediate size $= 11008$
* Sequence length $s = 2048$, Batch size $b = 2$

---

### Scenario A: Full Fine-Tuning (16-bit Mixed Precision AdamW)

In full fine-tuning, all $7.0 \times 10^9$ parameters are updated.

#### 1. Static Parameter Memory:
* **Model Weights (fp16/bf16):**
  $$M_{\text{weight}} = 7.0 \times 10^9 \times 2 \text{ bytes} = 14.0 \times 10^9 \text{ bytes} = \mathbf{14.00 \text{ GB}} \; (13.04 \text{ GiB})$$
* **Gradients (fp16/bf16):**
  $$M_{\text{grad}} = 7.0 \times 10^9 \times 2 \text{ bytes} = 14.0 \times 10^9 \text{ bytes} = \mathbf{14.00 \text{ GB}} \; (13.04 \text{ GiB})$$
* **Optimizer States (AdamW in fp32):**
  $$M_{\text{opt}} = 7.0 \times 10^9 \times (4_{\text{master}} + 4_{\text{momentum}} + 4_{\text{variance}}) \text{ bytes} = 7.0 \times 10^9 \times 12 = \mathbf{84.00 \text{ GB}} \; (78.23 \text{ GiB})$$
* **Total Static State:**
  $$M_{\text{static}} = 14.00 + 14.00 + 84.00 = \mathbf{112.00 \text{ GB}} \; (\mathbf{104.31 \text{ GiB}})$$

#### 2. Dynamic & Runtime Memory:
* **Activation Memory (with gradient checkpointing, $b=2, s=2048$):** $\approx \mathbf{3.50 \text{ GB}} \; (3.26 \text{ GiB})$
* **PyTorch CUDA Context & Temporary Buffers:** $\approx \mathbf{2.50 \text{ GB}} \; (2.33 \text{ GiB})$

#### 3. Total Peak Memory:
$$\text{Peak}_{\text{Full FT}} = 112.00 + 3.50 + 2.50 = \mathbf{118.00 \text{ GB}} \; (\mathbf{109.90 \text{ GiB}})$$

**Conclusion:** Cannot fit on any single commercial GPU. Requires at least **$2\times$ A100/H100 80GB GPUs** using DeepSpeed ZeRO-3 or PyTorch Fully Sharded Data Parallel (FSDP) to shard optimizer states across devices.

---

### Scenario B: LoRA (16-bit Frozen Base + Trainable Adapters)

In LoRA, base weights are frozen. Only low-rank matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times k}$ are trained.
Assuming rank $r=16$ targeting all linear projection layers (`q, k, v, o, gate, up, down_proj`):
* Trainable adapter parameters: $N_{\text{lora}} \approx 30.0 \times 10^6$ (0.43% of base model).

#### 1. Static Parameter Memory:
* **Frozen Base Model Weights (fp16/bf16):**
  $$M_{\text{base}} = 7.0 \times 10^9 \times 2 \text{ bytes} = \mathbf{14.00 \text{ GB}} \; (13.04 \text{ GiB})$$
* **Adapter Weights (fp16/bf16):**
  $$M_{\text{adapter}} = 30 \times 10^6 \times 2 \text{ bytes} = 0.06 \times 10^9 \text{ bytes} = \mathbf{0.06 \text{ GB}} \; (0.056 \text{ GiB})$$
* **Adapter Gradients (fp16/bf16):**
  $$M_{\text{adapter\_grad}} = 30 \times 10^6 \times 2 \text{ bytes} = \mathbf{0.06 \text{ GB}} \; (0.056 \text{ GiB})$$
* **Adapter Optimizer States (AdamW on trainable params only):**
  $$M_{\text{adapter\_opt}} = 30 \times 10^6 \times 12 \text{ bytes} = \mathbf{0.36 \text{ GB}} \; (0.335 \text{ GiB})$$
* **Total Static State:**
  $$M_{\text{static}} = 14.00 + 0.06 + 0.06 + 0.36 = \mathbf{14.48 \text{ GB}} \; (\mathbf{13.49 \text{ GiB}})$$

#### 2. Dynamic & Runtime Memory:
* **Activation Memory:** Base activations must still be maintained, plus slight adapter overhead: $\approx \mathbf{3.80 \text{ GB}} \; (3.54 \text{ GiB})$
* **PyTorch CUDA Context & Workspace:** $\approx \mathbf{2.00 \text{ GB}} \; (1.86 \text{ GiB})$

#### 3. Total Peak Memory:
$$\text{Peak}_{\text{LoRA}} = 14.48 + 3.80 + 2.00 = \mathbf{20.28 \text{ GB}} \; (\mathbf{18.89 \text{ GiB}})$$

**Conclusion:** Fits comfortably within a single **24 GB GPU** (NVIDIA RTX 3090, RTX 4090, A10G, or A5000).

---

### Scenario C: QLoRA (4-bit Quantized Base + 16-bit Adapters)

QLoRA replaces 16-bit frozen base weights with 4-bit NormalFloat4 (NF4) and employs Double Quantization (DQ) and Paged Optimizers.

#### 1. Static Parameter Memory:
* **4-bit Quantized Base Weights (NF4):**
  $$M_{\text{base\_4bit}} = 7.0 \times 10^9 \times 0.5 \text{ bytes} = \mathbf{3.50 \text{ GB}} \; (3.26 \text{ GiB})$$
* **Double Quantization (DQ) Metadata:**
  Quantization constants (FP32 block-level scale factors quantized to 8-bit FP8 with block size 64/256):
  $$M_{\text{DQ}} \approx 7.0 \times 10^9 \times 0.13 \text{ bytes} \approx \mathbf{0.91 \text{ GB}} \; (0.85 \text{ GiB})$$
  $$\text{Effective Base Weight Total} = 3.50 + 0.91 = \mathbf{4.41 \text{ GB}} \; (4.11 \text{ GiB})$$
* **Trainable LoRA Adapters (16-bit):** $\mathbf{0.06 \text{ GB}} \; (0.056 \text{ GiB})$
* **Adapter Gradients (16-bit):** $\mathbf{0.06 \text{ GB}} \; (0.056 \text{ GiB})$
* **Paged AdamW Optimizer (8-bit or 16-bit with CPU Paging):** $\approx \mathbf{0.18 \text{ GB}} \; (0.168 \text{ GiB})$
* **Total Static State:**
  $$M_{\text{static}} = 4.41 + 0.06 + 0.06 + 0.18 = \mathbf{4.71 \text{ GB}} \; (\mathbf{4.39 \text{ GiB}})$$

#### 2. Dynamic & Runtime Memory:
* **Activation Memory (with gradient checkpointing):**
  *(Activations are dequantized to 16-bit during the forward pass for matrix computation):* $\approx \mathbf{3.80 \text{ GB}} \; (3.54 \text{ GiB})$
* **PyTorch CUDA Context & Dequantization Buffers:** $\approx \mathbf{1.50 \text{ GB}} \; (1.40 \text{ GiB})$

#### 3. Total Peak Memory:
$$\text{Peak}_{\text{QLoRA}} = 4.71 + 3.80 + 1.50 = \mathbf{10.01 \text{ GB}} \; (\mathbf{9.32 \text{ GiB}})$$

**Conclusion:** Fits easily on entry-level consumer and cloud GPUs with **12 GB to 16 GB VRAM** (e.g., NVIDIA RTX 3060 12GB, RTX 4070 12GB, or free Google Colab T4 16GB).

---

## 4. Why Peak Memory Deviates in Practice

ML engineers cannot rely on static weight numbers alone. Real peak memory varies significantly due to 5 critical factors:

1. **Sequence Length Scaling ($s$):**
   * Attention matrix size scales quadratically $O(s^2)$ without FlashAttention.
   * With FlashAttention-2, peak attention memory scales linearly $O(s)$, but intermediate projection activations still scale directly with $s$. Increasing $s$ from 2,048 to 8,192 quadruples activation memory (+10 to 14 GiB).
2. **Batch Size ($b$) vs Gradient Accumulation:**
   * Increasing micro-batch size $b$ scales activation memory linearly.
   * **Best Practice:** Set micro-batch size $b=1$ or $b=2$, and achieve the target effective batch size (e.g., $b_{\text{eff}} = 64$) by setting `gradient_accumulation_steps = 32`. Gradient accumulation adds zero activation memory.
3. **Activation Checkpointing (Gradient Checkpointing):**
   * Disabling activation checkpointing increases memory by $4\times$ to $6\times$, causing immediate out-of-memory (OOM) errors even in LoRA.
4. **Attention Kernel Implementation:**
   * Native PyTorch eager attention allocates the full attention tensor ($b \times \text{heads} \times s \times s$).
   * FlashAttention-2 fuses the Softmax reduction into SRAM tiles, avoiding materializing the $s \times s$ attention map in DRAM entirely.
5. **Memory Fragmentation & CUDA Allocator Caching:**
   * PyTorch's caching allocator reserves memory blocks that are not immediately freed to avoid costly `cudaMalloc` calls.
   * Setting `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` reduces fragmentation spikes by up to 20%.
