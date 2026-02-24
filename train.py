"""
混合訓練腳本 - PrimeVul + Juliet Test Suite
使用 Focal Loss 處理類別不平衡
專注於 CWE-119/787 記憶體安全漏洞檢測
Intel XPU (Arc GPU) 優化版本
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
from tqdm import tqdm
import random

# 資料路徑
PRIMEVUL_TRAIN = "/home/nolan/下載/security-finder/primevul_train.jsonl"
PRIMEVUL_VALID = "/home/nolan/下載/security-finder/primevul_valid.jsonl"
JULIET_CSV = "/home/nolan/下載/2026_2_9/jts_c_1_3_train.csv"

OUTPUT_DIR = "./memsafety_focal_model"

MODEL_NAME = "microsoft/codebert-base"
BATCH_SIZE = 16
ACCUMULATION_STEPS = 2
EPOCHS = 8
LEARNING_RATE = 2e-5
MAX_LEN = 512
USE_BF16 = True

# Focal Loss 參數
FOCAL_GAMMA = 2.0    # 論文推薦值
FOCAL_ALPHA = 0.75   # 給少數類（漏洞）更高權重

TARGET_CWES = {'CWE-119', 'CWE-787'}
JULIET_TARGET_CWES = ["CWE119", "CWE787", "CWE121", "CWE122"]


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
    """
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.gamma
        alpha_weight = torch.where(targets == 1,
                                   torch.tensor(self.alpha, device=targets.device),
                                   torch.tensor(1 - self.alpha, device=targets.device))
        focal_loss = alpha_weight * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class PrimeVulDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len=512, mode='train', balance_ratio=1):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []

        print(f"\n[PrimeVul] 正在掃描: {file_path}")
        print(f"           目標 CWE: {TARGET_CWES}")
        print(f"           平衡比例: {balance_ratio}:1 (Safe:Vuln)")

        safe_samples = []
        vuln_samples = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Processing PrimeVul"):
                try:
                    data = json.loads(line)
                    target = int(data.get('target', 0))
                    func_code = data.get('func', '')
                    cwe_list = data.get('cwe', [])

                    if target == 0:
                        safe_samples.append((func_code, 0))
                    else:
                        if any(cwe in TARGET_CWES for cwe in cwe_list):
                            vuln_samples.append((func_code, 1))

                except Exception:
                    continue

        n_vuln = len(vuln_samples)

        if n_vuln == 0:
            print("⚠️ 警告：找不到符合 CWE-119/787 的漏洞樣本！")
        else:
            n_safe_keep = n_vuln * balance_ratio
            if len(safe_samples) > n_safe_keep:
                print(f"           平衡資料: 縮減 Safe {len(safe_samples)} -> {n_safe_keep}")
                random.shuffle(safe_samples)
                safe_samples = safe_samples[:n_safe_keep]

        self.samples = safe_samples + vuln_samples
        random.shuffle(self.samples)

        print(f"[PrimeVul] {mode.upper()} 準備完成:")
        print(f"           ✅ Safe (0): {len(safe_samples)}")
        print(f"           ❌ Vuln (1): {len(vuln_samples)}")
        print(f"           📊 總數: {len(self.samples)}")
        if len(self.samples) > 0:
            vuln_ratio = len(vuln_samples) / len(self.samples) * 100
            print(f"           🎯 漏洞佔比: {vuln_ratio:.1f}%")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        code, label = self.samples[idx]
        encoding = self.tokenizer.encode_plus(
            code,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class JulietMemSafetyDataset(Dataset):
    def __init__(self, tokenizer, csv_path, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.codes = []
        self.labels = []

        print(f"\n[Juliet] 載入資料集: {csv_path}")
        print(f"         目標 CWE: {JULIET_TARGET_CWES}")

        df = pd.read_csv(csv_path)
        df['class'] = df['class'].astype(str).fillna("")
        df['class'] = df['class'].apply(lambda x: f"CWE{x}" if x.isdigit() else x)

        print(f"[Juliet] 原始資料數量: {len(df)}")

        df = df[df["class"].apply(lambda x: any(x.startswith(t) for t in JULIET_TARGET_CWES))]

        print(f"[Juliet] Memory Safety 樣本數: {len(df)}")

        if len(df) == 0:
            print("⚠️ 警告: Juliet 沒有找到相關資料!")
            return

        for _, row in df.iterrows():
            good_code = row.get("good")
            if pd.notna(good_code) and isinstance(good_code, str) and good_code.strip():
                self.codes.append(good_code.strip())
                self.labels.append(0)

            bad_code = row.get("bad")
            if pd.notna(bad_code) and isinstance(bad_code, str) and bad_code.strip():
                self.codes.append(bad_code.strip())
                self.labels.append(1)

        self.labels = np.array(self.labels)
        self._print_statistics()

    def _print_statistics(self):
        total = len(self.labels)
        if total == 0:
            return
        vuln_count = np.sum(self.labels)
        safe_count = total - vuln_count

        print(f"[Juliet] 資料集統計:")
        print(f"         總計: {total:,} 個樣本")
        print(f"         ✅ Safe: {safe_count:,} ({safe_count/total*100:.1f}%)")
        print(f"         ❌ Vuln: {vuln_count:,} ({vuln_count/total*100:.1f}%)")

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.codes[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class MemSafetyDetector(nn.Module):
    def __init__(self, model_name, dropout=0.1):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(model_name)
        if hasattr(self.codebert, "gradient_checkpointing_enable"):
            self.codebert.gradient_checkpointing_enable()
        self.classifier = nn.Linear(self.codebert.config.hidden_size, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            use_autocast = (device.type == 'xpu' or device.type == 'cuda') and USE_BF16
            dtype = torch.bfloat16 if device.type == 'xpu' else torch.float16

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            val_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].float().cpu().numpy())

    avg_loss = val_loss / len(val_loader)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0

    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': avg_loss,
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cpu")
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        print("[+] 使用 Intel XPU (Arc GPU)")
        print("[+] ⚡ BF16 混合精度已啟用")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("[+] 使用 NVIDIA CUDA")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("\n" + "="*70)
    print("📚 載入混合資料集")
    print("="*70)
    print(f"🎯 Focal Loss 參數: alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA}")
    print("="*70)

    primevul_train = PrimeVulDataset(tokenizer, PRIMEVUL_TRAIN, MAX_LEN, mode='train', balance_ratio=1)
    primevul_valid = PrimeVulDataset(tokenizer, PRIMEVUL_VALID, MAX_LEN, mode='valid', balance_ratio=1)

    juliet_dataset = None
    if os.path.exists(JULIET_CSV):
        juliet_dataset = JulietMemSafetyDataset(tokenizer, JULIET_CSV, MAX_LEN)
        if len(juliet_dataset) == 0:
            juliet_dataset = None
    else:
        print(f"⚠️ 找不到 Juliet 檔案: {JULIET_CSV}")

    if juliet_dataset is not None and len(juliet_dataset) > 0:
        print(f"\n[混合] 合併 PrimeVul + Juliet")
        train_dataset = ConcatDataset([primevul_train, juliet_dataset])
        print(f"[混合] 總訓練樣本數: {len(train_dataset):,}")
    else:
        train_dataset = primevul_train

    valid_dataset = primevul_valid

    if len(train_dataset) == 0:
        print("❌ 錯誤：訓練集為空！")
        return

    print("\n" + "="*70)
    print(f"📊 最終資料集統計")
    print("="*70)
    print(f"   訓練集: {len(train_dataset):,} 樣本")
    print(f"   驗證集: {len(valid_dataset):,} 樣本")
    print("="*70)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = MemSafetyDetector(MODEL_NAME, dropout=0.1)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS // ACCUMULATION_STEPS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps*0.1),
        num_training_steps=total_steps
    )

    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)

    best_f1 = 0.0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_recall': [],
        'val_precision': []
    }

    print(f"\n🚀 開始訓練 (Focal Loss + Batch Balancing)")
    print(f"   Model: {MODEL_NAME}")
    print(f"   Batch Size: {BATCH_SIZE} x {ACCUMULATION_STEPS} = {BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Focal Loss: α={FOCAL_ALPHA}, γ={FOCAL_GAMMA}")
    print("="*70 + "\n")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for step, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            use_autocast = (device.type == 'xpu' or device.type == 'cuda') and USE_BF16
            dtype = torch.bfloat16 if device.type == 'xpu' else torch.float16

            with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_autocast):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / ACCUMULATION_STEPS

            loss.backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                if device.type == 'xpu':
                    torch.xpu.empty_cache()

            train_loss += loss.item() * ACCUMULATION_STEPS
            pbar.set_postfix({'loss': f"{train_loss / (step + 1):.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        print("\n驗證中...")
        metrics = validate(model, valid_loader, device, criterion)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(metrics['loss'])
        history['val_f1'].append(metrics['f1'])
        history['val_recall'].append(metrics['recall'])
        history['val_precision'].append(metrics['precision'])

        cm = metrics['confusion_matrix']

        print(f"\n{'='*70}")
        print(f"📊 Epoch {epoch+1}/{EPOCHS} 結果")
        print(f"{'='*70}")
        print(f"   訓練 Loss:  {avg_train_loss:.4f}")
        print(f"   驗證 Loss:  {metrics['loss']:.4f}")
        print(f"   Accuracy:   {metrics['accuracy']:.4f}")
        print(f"   Precision:  {metrics['precision']:.4f}")
        print(f"   Recall:     {metrics['recall']:.4f} ⭐")
        print(f"   F1-Score:   {metrics['f1']:.4f} 🎯")
        print(f"   AUC:        {metrics['auc']:.4f}")
        print(f"\n   混淆矩陣:")
        print(f"   ┌─────────────────────┐")
        print(f"   │  TN={cm[0][0]:4d}  FP={cm[0][1]:4d}  │")
        print(f"   │  FN={cm[1][0]:4d}  TP={cm[1][1]:4d}  │")
        print(f"   └─────────────────────┘")

        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"   Sensitivity (TPR): {sensitivity:.4f}")
        print(f"   Specificity (TNR): {specificity:.4f}")
        print(f"{'='*70}\n")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            tokenizer.save_pretrained(OUTPUT_DIR)

            info = {
                'epoch': epoch + 1,
                'best_f1': float(best_f1),
                'accuracy': float(metrics['accuracy']),
                'model_name': MODEL_NAME,
                'focal_alpha': FOCAL_ALPHA,
                'focal_gamma': FOCAL_GAMMA,
            }
            with open(os.path.join(OUTPUT_DIR, "training_info.json"), 'w') as f:
                json.dump(info, f, indent=2)
            print(f"💾 最佳模型已儲存！(F1: {best_f1:.4f})")

    with open(os.path.join(OUTPUT_DIR, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print("\n" + "="*70)
    print("✅ 訓練完成！")
    print(f"🏆 最佳 F1 分數: {best_f1:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()
