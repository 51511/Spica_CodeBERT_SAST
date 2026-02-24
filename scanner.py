"""
跨語言漏洞掃描器 (Universal Vulnerability Scanner)
基於 CodeBERT 預訓練模型架構
支援語意分析、跨語言特徵提取、長文本滑動視窗

適用範圍：
- C/C++ (專注於 Memory Safety: Buffer Overflow, OOB Write)
- Java/Python/Go (架構已支援，可透過遷移學習擴充)
- 支援 BF16 (Intel Arc/XPU) 硬體加速推論
"""

import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import time


class VulnerabilityDetector(nn.Module):
    def __init__(self, model_name="microsoft/codebert-base", dropout=0.1):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.codebert.config.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class CrossLanguageScanner:
    LANGUAGE_EXTENSIONS = {
        'C/C++': {'.c', '.cpp', '.cc', '.cxx', '.h', '.hpp', '.hxx'},
        'Java': {'.java'},
        'Python': {'.py'},
        'JavaScript': {'.js', '.ts'},
        'Go': {'.go'},
        'Ruby': {'.rb'},
    }

    def __init__(self, model_dir, threshold=0.5, target_cwes=None, use_bf16=True):
        self.device = self._setup_device()
        self.threshold = threshold
        self.max_len = 512
        self.use_bf16 = use_bf16 and self.device.type == 'xpu'
        self.target_cwes = target_cwes or []

        print(f"\n{'='*70}")
        print(f"🔍 跨語言漏洞掃描器")
        print(f"{'='*70}")
        print(f"[*] 載入模型: {model_dir}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.model = VulnerabilityDetector()
        model_path = os.path.join(model_dir, "best_model.pt")

        if not os.path.exists(model_path):
            alt_path = os.path.join(model_dir, "pytorch_model.bin")
            if os.path.exists(alt_path):
                model_path = alt_path
            else:
                raise FileNotFoundError(f"找不到模型檔案: {model_path}")

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        print(f"[+] 模型引擎啟動成功")
        print(f"[+] 運算裝置: {self.device}")
        if self.use_bf16:
            print(f"[+] ⚡ Intel XPU BF16 加速已啟用")
        print(f"[+] 檢測閾值: {threshold}")
        if self.target_cwes:
            print(f"[+] 目標 CWE: {', '.join(self.target_cwes)}")
        print(f"{'='*70}\n")

    def _setup_device(self):
        try:
            if hasattr(torch, 'xpu') and torch.xpu.is_available():
                return torch.device("xpu")
        except:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def detect_language(self, file_path):
        ext = Path(file_path).suffix.lower()
        for lang, extensions in self.LANGUAGE_EXTENSIONS.items():
            if ext in extensions:
                return lang
        return "Unknown"

    def scan_directory(self, target_dir, languages=None, batch_size=16, exclude_dirs=None):
        if exclude_dirs is None:
            exclude_dirs = {
                'node_modules', '.git', '__pycache__', 'venv',
                'dist', 'build', 'bin', 'obj', 'target', '.idea'
            }

        all_files = []
        language_stats = {}

        print(f"[*] 掃描目標: {target_dir}")

        if languages:
            target_extensions = set()
            for lang in languages:
                if lang in self.LANGUAGE_EXTENSIONS:
                    target_extensions.update(self.LANGUAGE_EXTENSIONS[lang])
        else:
            target_extensions = set()
            for exts in self.LANGUAGE_EXTENSIONS.values():
                target_extensions.update(exts)

        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for filename in files:
                ext = Path(filename).suffix.lower()
                if ext in target_extensions:
                    filepath = os.path.join(root, filename)
                    lang = self.detect_language(filepath)

                    all_files.append(filepath)
                    language_stats[lang] = language_stats.get(lang, 0) + 1

        print(f"\n📊 專案結構分析:")
        for lang, count in sorted(language_stats.items()):
            print(f"   {lang}: {count:,} 個檔案")
        print(f"   總計: {len(all_files):,} 個原始碼檔案\n")

        all_findings = []
        for filepath in tqdm(all_files, desc="深度掃描中", unit="file"):
            findings = self.scan_file_batch(filepath, batch_size)
            all_findings.extend(findings)

            if self.device.type == 'xpu' and len(all_findings) % 100 == 0:
                torch.xpu.empty_cache()

        return all_findings, language_stats

    def scan_file_batch(self, file_path, batch_size=16):
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
        except:
            return []

        chunks = self._split_code(code)
        if not chunks:
            return []

        findings = []
        language = self.detect_language(file_path)

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_results = self._batch_detect(batch_chunks)

            for idx, (is_vuln, confidence) in enumerate(batch_results):
                if is_vuln:
                    chunk_idx = i + idx
                    start_line, chunk_text = batch_chunks[idx]

                    finding = {
                        "file": file_path,
                        "language": language,
                        "line": start_line,
                        "confidence": confidence,
                        "snippet": chunk_text[:300].strip(),
                        "vulnerability": self._format_vulnerability_type(),
                        "severity": self._estimate_severity(confidence)
                    }
                    findings.append(finding)

        return findings

    def _batch_detect(self, chunk_list):
        codes = [chunk_text for _, chunk_text in chunk_list]

        inputs = self.tokenizer(
            codes,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            use_autocast = (self.device.type == 'xpu' or self.device.type == 'cuda') and self.use_bf16
            dtype = torch.bfloat16 if self.device.type == 'xpu' else torch.float16

            with torch.autocast(device_type=self.device.type, dtype=dtype, enabled=use_autocast):
                logits = self.model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask']
                )
                logits = logits.float()
                probs = torch.softmax(logits, dim=1)
                vuln_probs = probs[:, 1].cpu().numpy()

        results = []
        for prob in vuln_probs:
            is_vulnerable = prob >= self.threshold
            results.append((is_vulnerable, float(prob)))

        return results

    def _split_code(self, code, window=40, overlap=10):
        lines = code.split('\n')
        chunks = []

        for i in range(0, len(lines), window - overlap):
            chunk_lines = lines[i:i + window]
            if chunk_lines and any(line.strip() for line in chunk_lines):
                chunk_text = '\n'.join(chunk_lines)
                chunks.append((i + 1, chunk_text))

        return chunks

    def _format_vulnerability_type(self):
        if self.target_cwes:
            return f"檢測到異常特徵 ({', '.join(self.target_cwes)})"
        return "檢測到潛在漏洞特徵"

    def _estimate_severity(self, confidence):
        if confidence >= 0.9: return "Critical"
        elif confidence >= 0.75: return "High"
        elif confidence >= 0.6: return "Medium"
        else: return "Low"


def main():
    parser = argparse.ArgumentParser(
        description="跨語言漏洞掃描器 (基於 CodeBERT)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例：
  python scanner.py --scan_dir ./my_cpp_project --model_dir ./memsafety_focal_model --cwes CWE-119 CWE-787
  python scanner.py --scan_dir ./mixed_project --model_dir ./model --languages "C/C++" Java
  python scanner.py --scan_dir ./project --model_dir ./model --threshold 0.4
        """
    )

    parser.add_argument("--scan_dir", required=True, help="要掃描的專案目錄")
    parser.add_argument("--model_dir", required=True, help="訓練好的模型目錄")
    parser.add_argument("--threshold", type=float, default=0.5, help="檢測閾值 (預設 0.5)")
    parser.add_argument("--languages", nargs='+', help="指定掃描語言 (預設: 自動偵測)")
    parser.add_argument("--batch_size", type=int, default=16, help="推論批次大小")
    parser.add_argument("--no-bf16", action='store_true', help="停用 BF16 硬體加速")
    parser.add_argument("--cwes", nargs='+', default=["CWE-119", "CWE-787"], help="目標 CWE 標籤")

    args = parser.parse_args()

    if not os.path.exists(args.model_dir):
        print(f"❌ 錯誤：找不到模型目錄 {args.model_dir}")
        exit(1)

    if not os.path.exists(args.scan_dir):
        print(f"❌ 錯誤：找不到掃描目錄 {args.scan_dir}")
        exit(1)

    scanner = CrossLanguageScanner(
        model_dir=args.model_dir,
        threshold=args.threshold,
        target_cwes=args.cwes,
        use_bf16=not args.no_bf16
    )

    start_time = time.time()
    findings, language_stats = scanner.scan_directory(
        target_dir=args.scan_dir,
        languages=args.languages,
        batch_size=args.batch_size
    )
    elapsed = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"📊 掃描完成報告")
    print(f"{'='*70}")
    print(f"⏱️  耗時: {elapsed:.2f} 秒")
    print(f"📁 檔案總數: {sum(language_stats.values()):,} (涵蓋 {len(language_stats)} 種語言)")
    print(f"🔥 發現風險: {len(findings)} 處")
    print(f"{'='*70}\n")

    if findings:
        findings_sorted = sorted(findings, key=lambda x: x['confidence'], reverse=True)

        print("🔝 高風險項目預覽:\n")
        for i, finding in enumerate(findings_sorted[:10], 1):
            print(f"{i}. [{finding['severity']}] {Path(finding['file']).name} (Line {finding['line']})")
            print(f"   語言: {finding['language']}")
            print(f"   信心度: {finding['confidence']:.2%}")
            print(f"   程式碼: {finding['snippet'].splitlines()[0].strip()}...")
            print()

        report_path = "scan_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump({'findings': findings}, f, indent=2, ensure_ascii=False)
        print(f"💾 完整報告已儲存至: {report_path}")

    else:
        print("✅ 未檢測到高風險特徵")


if __name__ == "__main__":
    main()
