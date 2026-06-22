#!/usr/bin/env python3
"""端到端資料流編排器 (prediction -> decision)。

用「一個指令」跑完整條鏈：

    STGCN/run_infer.py ─┐
                        ├─> integration/stg{cn,at}_pred.npy   (預測)
    STGAT/run_infer.py ─┘
                        └─> run_compare.py                    (路由決策比較)

為什麼推論階段用 subprocess 而不是直接 import：
STGCN 與 STGAT 兩個 repo 各自有一個頂層 `model` package (`model.models` vs
`model.stgat`)，import 進同一個 process 會互相覆蓋。讓每個 repo 用自己的工作
目錄各自執行 run_infer.py，可完全隔離、可重現。

決策階段 (run_compare.py) 與所有核心模組 (config / network / policies / metrics)
都在「本資料夾內」，不跨目錄相依。

用法:
    python pipeline.py                # 完整流程 (重跑推論，需要 GPU)
    python pipeline.py --skip-infer   # 重用既有的 *_pred.npy (快、免 GPU)
    python pipeline.py --skip-infer --scenario random   # 額外參數轉傳給 run_compare.py
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent     # .../integration
ROOT = HERE.parent                          # 專案根目錄

# (說明, 工作目錄, 腳本) — 推論階段，在各自的 model repo 內執行
INFER_STEPS = [
    ("STGCN inference", ROOT / "STGCN", "run_infer.py"),
    ("STGAT inference", ROOT / "STGAT", "run_infer.py"),
]
PRED_FILES = [HERE / "stgcn_pred.npy", HERE / "stgat_pred.npy"]


def run(label, args, cwd):
    print(f"\n{'=' * 64}\n▶ {label}\n  ({cwd}$ {' '.join(args)})\n{'=' * 64}")
    t0 = time.time()
    proc = subprocess.run([sys.executable, *args], cwd=str(cwd))
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"✗ 「{label}」失敗 (exit {proc.returncode})，耗時 {dt:.1f}s")
        sys.exit(proc.returncode)
    print(f"✓ 「{label}」完成 ({dt:.1f}s)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-infer", action="store_true",
                    help="重用既有的 *_pred.npy，不重跑模型推論")
    # 其餘未知參數原樣轉傳給 run_compare.py (例如 --scenario / --vehicles / --capacity)
    args, passthrough = ap.parse_known_args()

    if args.skip_infer:
        missing = [p.name for p in PRED_FILES if not p.exists()]
        if missing:
            sys.exit(f"--skip-infer 已設定，但缺少預測檔: {missing}。"
                     f"請先不加 --skip-infer 跑一次。")
        print("• 跳過推論，重用既有預測:", ", ".join(p.name for p in PRED_FILES))
    else:
        for label, cwd, script in INFER_STEPS:
            run(label, [script], cwd)

    run("Routing comparison", ["run_compare.py", *passthrough], HERE)

    print(f"\n{'=' * 64}\n✓ Pipeline 完成。輸出位於 {HERE}\n{'=' * 64}")


if __name__ == "__main__":
    main()
