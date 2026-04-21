# CodeBERT Memory Safety Vulnerability Detector

A lightweight automated detection system for C/C++ memory safety vulnerabilities, built on CodeBERT and Focal Loss.

> Independent research by a high school student. Trained entirely on a consumer-grade Intel Arc A380 GPU (6GB VRAM).

## Results & Performance

Evaluated on the **CASTLE-C250** benchmark targeting memory safety vulnerabilities (CWE-119 / CWE-787). 

To reflect real-world engineering scenarios, the metrics below are calculated using a strict **File-Level Evaluation** (if any chunk within a file triggers an alert, the entire file is flagged), rather than a naive chunk-level metric.

| Model | Recall | FPR (False Positive Rate) | F1-Score |
|-------|--------|---------------------------|----------|
| CodeBERT (Baseline) | 11.94% | - | 0.119 |
| CodeT5 (SOTA) | 17.21% | - | 0.172 |
| **This Work (Focal Loss, Threshold=0.5)** | **57.50%** | **60.95%** | **0.241** |

### Key Findings & Contributions

- **Balancing Recall and FPR**: By integrating Focal Loss (α=0.75, γ=2.0) to address severe class imbalance in security data, the model avoids the common pitfall of zero-shot models either missing everything or flagging everything. It achieves a highly competitive **57.50% Recall** (a massive improvement over baseline models) while constraining the False Positive Rate to 60.95%.
- **Cross-Boundary Context**: Implementing a sliding window mechanism (window: 40 lines, overlap: 10 lines) effectively circumvents CodeBERT's native 512-token limit, allowing the model to capture vulnerabilities spanning multiple code blocks.
- **Zero-Shot Generalization**: An unexpected finding during evaluation was the model's ability to generalize beyond its training distribution. It successfully detected CWE-327, CWE-770, and CWE-798 vulnerabilities despite never being explicitly trained on them—suggesting CodeBERT's pre-training combined with Focal Loss encodes highly generalizable unsafe code semantics.

## Architecture

- **Backbone**: Microsoft CodeBERT (`codebert-base`, 125M parameters)
- **Loss Function**: Focal Loss — significantly improves the discriminative power on hard-to-classify, minority vulnerability samples.
- **Input Handling**: Overlapping Sliding Window
- **Training Data**: Hybrid strategy combining PrimeVul (real-world) + Juliet Test Suite (synthetic)
- **Hardware Acceleration**: Intel Arc A380 (6GB VRAM), utilizing BF16 mixed precision via Intel Extension for PyTorch (IPEX).

## Research Context

This work targets **CWE-119 (Buffer Overflow)** and **CWE-787 (Out-of-Bounds Write)** — two of the most prevalent memory safety vulnerability classes in C/C++. According to Microsoft MSRC and Google Chromium, ~70% of high-severity CVEs are memory safety issues.

Training uses **PrimeVul** — the same dataset utilized by Google DeepMind's Gemini 1.5 Pro for vulnerability detection evaluation — combined with **Juliet Test Suite v1.3** (NIST SARD).

> *Note: Model weights and training data are not included in this repository. The codebase demonstrates the full training and inference pipeline. Reproducing results requires independent data preparation and benchmark setup.*

## Requirements

```text
torch
transformers
scikit-learn
pandas
numpy
tqdm
intel-extension-for-pytorch  # optional, required for Intel XPU (BF16) acceleration
```

## Usage

### 1. Training

Edit data paths in `train.py`, then run:

```bash
python train.py
```
*Model checkpoint will be saved to `./memsafety_focal_model/`.*

### 2. Scanning a Project

```bash
# Scan a C/C++ codebase with default threshold
python scanner.py --scan_dir ./your_project --model_dir ./memsafety_focal_model

# Adjust detection threshold (e.g., lower to 0.5 for higher recall)
python scanner.py --scan_dir ./your_project --model_dir ./memsafety_focal_model --threshold 0.5
```

## Datasets

- **PrimeVul**: Real-world vulnerability dataset extracted from open-source projects, covering 140+ CWE types.
- **Juliet Test Suite v1.3**: Synthetic vulnerability benchmark developed by NIST, covering 118 CWE types.

## Reference

If you find this work relevant to your research, please feel free to reach out. Full research report (25 pages) available upon request.

## License

GPL v3

---
