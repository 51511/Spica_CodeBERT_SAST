# CodeBERT Memory Safety Vulnerability Detector

基於 CodeBERT 與 Focal Loss 的 C/C++ 記憶體安全漏洞自動偵測系統。

## 研究摘要

針對 CWE-119（緩衝區溢位）與 CWE-787（越界寫入）設計的輕量化漏洞偵測模型。透過混合訓練策略結合合成資料（Juliet Test Suite）與真實漏洞資料（PrimeVul），並引入 Focal Loss 解決資安資料樣本不平衡問題。

在 CASTLE-C250 基準測試上達到 **76.67% Recall**，相較 CodeBERT baseline（11.94%）提升 4.4 倍。

| 模型 | Recall | F1 |
|------|--------|----|
| CodeBERT (Baseline) | 11.94% | 0.119 |
| CodeT5 | 17.21% | 0.172 |
| **本研究 (Focal Loss)** | **76.67%** | **0.254** |

## 系統架構

- **骨幹模型**：Microsoft CodeBERT (`codebert-base`, 125M 參數)
- **損失函數**：Focal Loss（α=0.75, γ=2.0）
- **輸入處理**：滑動視窗機制（視窗大小 40 行，重疊 10 行）
- **訓練資料**：PrimeVul + Juliet Test Suite 混合訓練
- **硬體加速**：Intel XPU BF16 混合精度

## 環境需求

```
torch
transformers
scikit-learn
pandas
numpy
tqdm
intel-extension-for-pytorch  # Intel XPU 加速（可選）
```

## 使用方式

### 訓練模型

修改 `train.py` 中的資料路徑後執行：

```bash
python train.py
```

訓練完成後模型儲存於 `./memsafety_focal_model/`。

### 掃描專案

```bash
# 掃描 C/C++ 專案
python scanner.py --scan_dir ./your_project --model_dir ./memsafety_focal_model

# 指定閾值
python scanner.py --scan_dir ./your_project --model_dir ./memsafety_focal_model --threshold 0.4
```

## 資料集

- **PrimeVul**：真實開源專案漏洞資料集，涵蓋 140+ CWE 類型
- **Juliet Test Suite v1.3**：NIST 開發的合成漏洞測試集，涵蓋 118 種 CWE

## License

GPL v3
