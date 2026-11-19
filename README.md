# Tndrr AI Internship Assignment

This repository contains the assignments and projects completed for the **Tndrr AI Internship**.

---

## 📁 Repository Structure

* **[`task-3/`](task-3/)**: **Running an Open LLM Locally & Optimization**
  * Hardware: NVIDIA GeForce RTX 2050 (4 GB VRAM) + AMD Ryzen 7 5800HS + 16 GB RAM.
  * Model: Meta Llama 3.2 3B Instruct evaluated across 4-bit (`Q4_K_M`) vs. 8-bit (`Q8_0`) quantization.
  * Measured Performance: **39.43 tokens/sec (4-bit)** vs. **14.18 tokens/sec (8-bit)** (2.78x speedup).
  * Contains automated benchmarking harness, raw JSON telemetry, 10-prompt qualitative comparison, KV cache / quantization theory, and 20-user 24 GB GPU capacity plan.
  * See the full report in [`task-3/TASK_3_FINAL_REPORT.md`](task-3/TASK_3_FINAL_REPORT.md).
