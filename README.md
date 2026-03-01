# CodeBERT Memory Safety Vulnerability Detector

A lightweight automated detection system for C/C++ memory safety vulnerabilities, built on CodeBERT and Focal Loss.

> Independent research by a high school student. Trained entirely on a consumer-grade Intel Arc A380 GPU (6GB VRAM).

## Results

Evaluated on the **CASTLE-C250** benchmark targeting buffer overflow vulnerabilities (CWE-119/787):

| Model | Recall | F1 |
|-------|--------|----|
| CodeBERT (Baseline) | 11.94% | 0.119 |
| CodeT5 (SOTA) | 17.21% | 0.172 |
| **This Work (Focal Loss)** | **76.67%** | **0.254** |

**4.4x Recall improvement over CodeBERT baseline — without graph-based structural augmentation.**

An unexpected finding: the model generalizes beyond its training distribution, successfully detecting CWE-327, CWE-770, and CWE-798 vulnerabilities despite never being trained on them — suggesting CodeBERT's pre-training encodes generalizable unsafe code semantics.

## Architecture

- **Backbone**: Microsoft CodeBERT (`codebert-base`, 125M parameters)
- **Loss Function**: Focal Loss (α=0.75, γ=2.0) — addresses severe class imbalance in security data
- **Input Handling**: Sliding window mechanism (window: 40 lines, overlap: 10 lines) — preserves cross-boundary context beyond CodeBERT's 512-token limit
- **Training Data**: Hybrid strategy combining PrimeVul (real-world) + Juliet Test Suite (synthetic)
- **Hardware**: Intel Arc A380 (6GB VRAM), BF16 mixed precision via Intel Extension for PyTorch

## Research Context

This work targets **CWE-119 (Buffer Overflow)** and **CWE-787 (Out-of-Bounds Write)** — two of the most prevalent memory safety vulnerability classes in C/C++. According to Microsoft MSRC and Google Chromium, ~70% of high-severity CVEs are memory safety issues.

Training uses **PrimeVul** — the same dataset used by Google DeepMind's Gemini 1.5 Pro for vulnerability detection evaluation — combined with **Juliet Test Suite v1.3** (NIST SARD).

> Note: Model weights and training data are not included in this repository. The codebase demonstrates the full training and inference pipeline. Reproducing results requires independent data preparation and benchmark setup.

## Requirements

```
torch
transformers
scikit-learn
pandas
numpy
tqdm
intel-extension-for-pytorch  # optional, for Intel XPU acceleration
```

## Usage

### Training

Edit data paths in `train.py`, then run:

```bash
python train.py
```

Model checkpoint saved to `./memsafety_focal_model/`.

### Scanning a Project

```bash
# Scan a C/C++ codebase
python scanner.py --scan_dir ./your_project --model_dir ./memsafety_focal_model

# Adjust detection threshold
python scanner.py --scan_dir ./your_project --model_dir ./memsafety_focal_model --threshold 0.4
```

## Datasets

- **PrimeVul**: Real-world vulnerability dataset extracted from open-source projects, covering 140+ CWE types
- **Juliet Test Suite v1.3**: Synthetic vulnerability benchmark developed by NIST, covering 118 CWE types

## Reference

If you find this work relevant to your research, please feel free to reach out.

Full research report (25 pages) available upon request.

## License

GPL v3


Chen, N. (2026). Lightweight Memory Safety Vulnerability Detection via CodeBERT and Focal Loss. Zenodo. https://doi.org/10.5281/zenodo.18816400
