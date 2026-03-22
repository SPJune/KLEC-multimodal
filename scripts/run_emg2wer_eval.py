#!/usr/bin/env python3
"""
각 exp_name에 대해:
1. 3개 epoch 체크포인트에 대해 valid split으로 emg2wer.sh 실행
2. CER이 가장 낮은 체크포인트 선택
3. 해당 체크포인트로 valid, test 모두 실행 후 results.csv에 저장
"""
import argparse
import csv
import os
import re
import subprocess
import sys
from glob import glob
from pathlib import Path

# 프로젝트 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_exp_path():
    """configs/common.yaml에서 exp_path 로드"""
    try:
        from omegaconf import OmegaConf
        common_path = PROJECT_ROOT / "configs" / "common.yaml"
        if common_path.exists():
            cfg = OmegaConf.load(common_path)
            return cfg.get("exp_path", "/data2/spjune/exp_ss_klec")
    except Exception:
        pass
    return "/data2/spjune/exp_ss_klec"


def get_epoch_checkpoints(exp_path: str, exp_name: str, max_epochs: int = 3) -> list[int]:
    """
    exp_name 디렉터리에서 epoch 체크포인트 목록 반환 (last.ckpt 제외).
    new-style: XX-*.ckpt, legacy: epoch=XX*
    최대 max_epochs개까지 반환 (에폭 번호 순 정렬).
    """
    exp_dir = Path(exp_path) / exp_name
    if not exp_dir.is_dir():
        return []

    epochs = set()
    # new-style: 05-0.9234.ckpt
    for p in glob(str(exp_dir / "[0-9]*-*.ckpt")):
        base = os.path.basename(p)
        if "epoch=" in base:
            continue
        m = re.match(r"^(\d+)-", base)
        if m:
            epochs.add(int(m.group(1)))
    # legacy: epoch=05*.ckpt
    for p in glob(str(exp_dir / "epoch=[0-9]*")):
        base = os.path.basename(p)
        m = re.search(r"epoch=(\d+)", base)
        if m:
            epochs.add(int(m.group(1)))

    sorted_epochs = sorted(epochs)[:max_epochs]
    return sorted_epochs


def run_emg2wer(exp_name: str, ckpt_epoch: int, split: str, gpu: int = 0) -> subprocess.CompletedProcess:
    """emg2wer.sh 실행"""
    script_path = PROJECT_ROOT / "infer" / "emg2wer.sh"
    cmd = ["bash", str(script_path), exp_name, str(ckpt_epoch), split, str(gpu)]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)


def parse_cer_wer(output: str) -> tuple[float, float, float] | None:
    """
    "CER: 27.58%, CER without space: 28.14%, WER: 66.23%" 파싱
    Returns (cer, cer_wo_space, wer) as percentages (0-100), or None if not found.
    """
    patterns = [
        # 기본: 한 줄
        r"CER:\s*([\d.]+)%,\s*CER without space:\s*([\d.]+)%,\s*WER:\s*([\d.]+)%",
        # 여러 줄/공백 허용
        r"CER:\s*([\d.]+)%[\s\S]*?CER without space:\s*([\d.]+)%[\s\S]*?WER:\s*([\d.]+)%",
        # CER만 따로 (순서 보장)
        r"CER:\s*([\d.]+)%(?!\d)",
        r"CER without space:\s*([\d.]+)%(?!\d)",
        r"WER:\s*([\d.]+)%(?!\d)",
    ]
    m = re.search(patterns[0], output)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    m = re.search(patterns[1], output, re.DOTALL)
    if m:
        return float(m.group(1)), float(m.group(2)), float(m.group(3))
    # 개별 숫자 추출 (마지막 등장 값 사용 - 보통 요약이 마지막에 옴)
    cer_m = list(re.finditer(r"CER:\s*([\d.]+)%", output))
    cer_wo_m = list(re.finditer(r"CER without space:\s*([\d.]+)%", output))
    wer_m = list(re.finditer(r"WER:\s*([\d.]+)%", output))
    if cer_m and cer_wo_m and wer_m:
        return float(cer_m[-1].group(1)), float(cer_wo_m[-1].group(1)), float(wer_m[-1].group(1))
    return None


def try_parse_from_log(exp_path: str, exp_name: str, split: str) -> tuple[float, float, float] | None:
    """asr_whisper.log 파일에서 CER/WER 파싱 (fallback)"""
    log_path = Path(exp_path) / exp_name / "direct" / split / "asr_whisper.log"
    if not log_path.exists():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        return parse_cer_wer(text)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Run emg2wer evaluation and save results to CSV")
    parser.add_argument("exp_names", nargs="+", help="Experiment names to evaluate")
    parser.add_argument("--exp-path", default=None, help="Experiment root path (default: from configs/common.yaml)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--max-epochs", type=int, default=3, help="Max epoch checkpoints to evaluate per exp (default: 3)")
    parser.add_argument("--output", default="results.csv", help="Output CSV path")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument("--debug", action="store_true", help="Print subprocess output when parse fails")
    args = parser.parse_args()

    exp_path = args.exp_path or load_exp_path()
    rows = []

    for exp_name in args.exp_names:
        epochs = get_epoch_checkpoints(exp_path, exp_name, max_epochs=args.max_epochs)
        if not epochs:
            print(f"[WARN] No epoch checkpoints found for {exp_name}, skipping.")
            continue

        print(f"\n=== {exp_name} (epochs: {epochs}) ===")
        valid_results = {}  # ckpt_epoch -> (cer, cer_wo_space, wer)

        for ckpt_epoch in epochs:
            print(f"  Running valid for ckpt_epoch={ckpt_epoch}...")
            if args.dry_run:
                print(f"    [DRY] bash infer/emg2wer.sh {exp_name} {ckpt_epoch} valid {args.gpu}")
                valid_results[ckpt_epoch] = (0.0, 0.0, 0.0)  # placeholder
                continue

            proc = run_emg2wer(exp_name, ckpt_epoch, "valid", args.gpu)
            combined = proc.stdout + "\n" + proc.stderr
            parsed = parse_cer_wer(combined)
            if parsed is None:
                parsed = try_parse_from_log(exp_path, exp_name, "valid")
            if parsed is None:
                print(f"    [ERR] Could not parse CER/WER from output.")
                if proc.returncode != 0:
                    print(f"    stderr (last 500): {proc.stderr[-500:]}")
                if args.debug:
                    print(f"    [DEBUG] stdout (last 1500): {combined[-1500:]}")
                continue
            cer, cer_wo_space, wer = parsed
            valid_results[ckpt_epoch] = (cer, cer_wo_space, wer)
            print(f"    CER: {cer:.2f}%, CER w/o space: {cer_wo_space:.2f}%, WER: {wer:.2f}%")

        if not valid_results:
            continue

        best_epoch = min(valid_results.keys(), key=lambda e: valid_results[e][0])
        best_cer, best_cer_wo, best_wer = valid_results[best_epoch]
        print(f"  Best: ckpt_epoch={best_epoch} (CER={best_cer:.2f}%)")

        # valid 결과 저장
        rows.append({
            "exp_name": exp_name,
            "ckpt_epoch": best_epoch,
            "split": "valid",
            "cer": f"{best_cer:.2f}",
            "cer_wo_space": f"{best_cer_wo:.2f}",
            "wer": f"{best_wer:.2f}",
        })

        # test 실행
        print(f"  Running test for ckpt_epoch={best_epoch}...")
        if args.dry_run:
            print(f"    [DRY] bash infer/emg2wer.sh {exp_name} {best_epoch} test {args.gpu}")
            rows.append({
                "exp_name": exp_name,
                "ckpt_epoch": best_epoch,
                "split": "test",
                "cer": "N/A",
                "cer_wo_space": "N/A",
                "wer": "N/A",
            })
        else:
            proc = run_emg2wer(exp_name, best_epoch, "test", args.gpu)
            combined = proc.stdout + "\n" + proc.stderr
            parsed = parse_cer_wer(combined)
            if parsed is None:
                parsed = try_parse_from_log(exp_path, exp_name, "test")
            if parsed is None:
                print(f"    [ERR] Could not parse CER/WER from test output.")
                if args.debug:
                    print(f"    [DEBUG] stdout+stderr (last 1500): {combined[-1500:]}")
                rows.append({
                    "exp_name": exp_name,
                    "ckpt_epoch": best_epoch,
                    "split": "test",
                    "cer": "N/A",
                    "cer_wo_space": "N/A",
                    "wer": "N/A",
                })
            else:
                cer_t, cer_wo_t, wer_t = parsed
                rows.append({
                    "exp_name": exp_name,
                    "ckpt_epoch": best_epoch,
                    "split": "test",
                    "cer": f"{cer_t:.2f}",
                    "cer_wo_space": f"{cer_wo_t:.2f}",
                    "wer": f"{wer_t:.2f}",
                })
                print(f"    CER: {cer_t:.2f}%, CER w/o space: {cer_wo_t:.2f}%, WER: {wer_t:.2f}%")

    # CSV 저장
    out_path = Path(args.output)
    if out_path.is_absolute():
        csv_path = out_path
    else:
        csv_path = PROJECT_ROOT / out_path

    fieldnames = ["exp_name", "ckpt_epoch", "split", "cer", "cer_wo_space", "wer"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"\nResults saved to {csv_path}")


if __name__ == "__main__":
    main()
