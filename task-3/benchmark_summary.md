# LLM Benchmark Experiment Summary

## Overall Average Metrics

| Configuration | Quantisation | Avg Generation Speed (tok/s) | Avg Prompt Eval Speed (tok/s) | Initial Load Time (s) | Peak VRAM (MiB) | Peak RAM (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Llama 3.2 3B (4-bit Q4_K_M) | 4-bit (Q4_K_M) | 39.43 | 343.5 | 6.1443 | 2327.0 | 10780.29 |
| Llama 3.2 3B (8-bit Q8_0) | 8-bit (Q8_0) | 14.18 | 221.18 | 7.3117 | 2351.0 | 11792.09 |


## Per-Prompt Detailed Metrics

| Prompt # | Category | 4-bit Gen (tok/s) | 8-bit Gen (tok/s) | Speed Ratio (4-bit / 8-bit) |
| :---: | :--- | :---: | :---: | :---: |
| 1 | General Knowledge | 38.83 | 13.45 | 2.89x |
| 2 | Reasoning | 39.37 | 14.26 | 2.76x |
| 3 | Mathematics | 39.31 | 13.88 | 2.83x |
| 4 | Summarisation | 40.11 | 14.65 | 2.74x |
| 5 | Instruction Following | 39.64 | 15.22 | 2.6x |
| 6 | Coding | 39.97 | 14.64 | 2.73x |
| 7 | Creative Writing | 39.31 | 13.58 | 2.89x |
| 8 | Information Extraction | 39.84 | 14.46 | 2.76x |
| 9 | Factual Accuracy | 39.4 | 14.17 | 2.78x |
| 10 | Challenging Multi-Step Task | 38.54 | 13.44 | 2.87x |
