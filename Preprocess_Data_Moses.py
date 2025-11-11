# =============================================================
#  preprocess_data_moses.py  (v1.2)
#  ──────────────────────────────────────────────────────────────
#  將 /mnt/g/_python/.../_dataset 內的 Moses 檔 (*.en / *.ja / *.zh / *.zh_tw)
#  自動配對→雙向擴增 prompt→tokenize→儲存為 Arrow：
#      qwen3_cje_lora/tokenized_<dataset_name>
#
#  ★ 更新重點 (v1.2)
#    • --data_root 改為「可選」參數，預設路徑為
#        /mnt/g/_python/Qwen3 1.7B finetune/_dataset
#    • 其餘功能不變
#
#  用法：
#    1. 採用預設資料夾
#       $ python preprocess_data_moses.py
#
#    2. 指定其他資料夾
#       $ python preprocess_data_moses.py --data_root /path/to/moses
# =============================================================

from __future__ import annotations
import os, gc, argparse, itertools, json, warnings
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------
#  CONFIGURATION
# -------------------------------------------------------------
DEFAULT_DATA_ROOT = "/mnt/g/_python/Qwen3 1.7B finetune/_dataset"
MODEL_NAME_OR_PATH = os.getenv(
    "QWEN_MODEL",
    "/mnt/g/_python/Qwen3 1.7B finetune/_Models/Qwen3-1.7B",
)
OUTPUT_ROOT = Path("qwen3_cje_lora")
MAX_SEQ_LEN = 2048
CHUNK_LINES = 250_000  # 讀檔分塊，避免佔用過多 RAM

# 語言標籤與副檔名對應
LANG_TAGS = ["<zh_TW>", "<en>", "<ja>"]
DEFAULT_LANG_MAP: Dict[str, str] = {
    "en": "en",
    "ja": "ja",
    "zh": "zh_tw",
    "zh_tw": "zh_tw",
    "zh_hant": "zh_tw",
}

# -------------------------------------------------------------
#  TOKENIZER & PROMPT UTILITIES
# -------------------------------------------------------------

def ensure_tokenizer_special_tokens(tokenizer):
    """確保 tokenizer 內含語言特殊 token"""
    tokenizer.add_special_tokens({"additional_special_tokens": LANG_TAGS})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def build_prompt(src_tag: str, tgt_tag: str, src: str, tgt: str) -> str:
    """組成 Qwen3 chat 格式 prompt (system+user+assistant)"""
    return (
        "<|im_start|>system\nYou are a precise and helpful multilingual translation assistant.<|im_end|>\n"
        f"<|im_start|>user\n{src_tag} {tgt_tag}\n{src}<|im_end|>\n"
        f"<|im_start|>assistant\n{tgt}<|im_end|>"
    )

# -------------------------------------------------------------
#  DISCOVER & PROCESS FILES
# -------------------------------------------------------------

def discover_parallel_files(root: Path, lang_map: Dict[str, str]) -> List[Tuple[Path, Path, str, str]]:
    """在 root 內找出平行檔對，回傳 (src_path, tgt_path, src_code, tgt_code) 列表"""
    clusters: Dict[str, Dict[str, Path]] = {}
    for path in root.rglob("*.*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lstrip(".")
        if suffix not in lang_map:
            continue
        stem = ".".join(path.name.split(".")[:-1])
        key = str(path.parent / stem)
        clusters.setdefault(key, {})[suffix] = path

    pairs = []
    for files in clusters.values():
        langs = sorted(files.keys())
        if len(langs) < 2:
            continue
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                l1, l2 = langs[i], langs[j]
                pairs.append((files[l1], files[l2], lang_map[l1], lang_map[l2]))
    return pairs


def preprocess_pair(src_path: Path, tgt_path: Path, src_code: str, tgt_code: str, tokenizer):
    dataset_name = f"{src_path.stem.split('.')[0]}_{src_code}-{tgt_code}"
    out_dir = OUTPUT_ROOT / f"tokenized_{dataset_name}"
    if out_dir.exists():
        print(f"[skip] {dataset_name}")
        return

    token_chunks = []
    with src_path.open(encoding="utf-8") as f_src, tgt_path.open(encoding="utf-8") as f_tgt:
        while True:
            src_chunk = list(itertools.islice(f_src, CHUNK_LINES))
            tgt_chunk = list(itertools.islice(f_tgt, CHUNK_LINES))
            if not src_chunk or not tgt_chunk:
                break
            n = min(len(src_chunk), len(tgt_chunk))
            src_chunk, tgt_chunk = src_chunk[:n], tgt_chunk[:n]

            prompts = [
                build_prompt(f"<{src_code}>", f"<{tgt_code}>", s.strip(), t.strip())
                for s, t in zip(src_chunk, tgt_chunk) if s.strip() and t.strip()
            ] + [
                build_prompt(f"<{tgt_code}>", f"<{src_code}>", t.strip(), s.strip())
                for s, t in zip(src_chunk, tgt_chunk) if s.strip() and t.strip()
            ]

            ds = Dataset.from_dict({"text": prompts})
            ds_tok = ds.map(
                lambda x: tokenizer(x["text"], truncation=True, max_length=MAX_SEQ_LEN, padding=False),
                batched=True,
                num_proc=1,
                remove_columns=["text"],
            )
            token_chunks.append(ds_tok)
            del ds, ds_tok, prompts
            gc.collect()

    if token_chunks:
        merged = concatenate_datasets(token_chunks)
        out_dir.mkdir(parents=True, exist_ok=True)
        merged.save_to_disk(out_dir)
        print(f"[saved] {dataset_name}: {len(merged):,} samples → {out_dir}")

# -------------------------------------------------------------
#  CLI ENTRY
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Preprocess Moses parallel data for Qwen3 fine‑tuning")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT,
                        help=f"Moses 資料夾母路徑 (預設: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--lang_map", help="JSON 字串覆寫副檔名→語言代碼，如 '{\"zht\":\"zh_tw\"}'")
    args = parser.parse_args()

    lang_map = DEFAULT_LANG_MAP.copy()
    if args.lang_map:
        lang_map.update(json.loads(args.lang_map))

    data_root = Path(args.data_root).expanduser().resolve()
    print(f"↳ 將在 {data_root} 掃描 Moses 檔案…")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    ensure_tokenizer_special_tokens(tokenizer)
    (OUTPUT_ROOT / "tokenizer").mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(OUTPUT_ROOT / "tokenizer")

    pairs = discover_parallel_files(data_root, lang_map)
    print(f"✔ 找到 {len(pairs)} 組平行檔對")

    for src, tgt, s_code, t_code in pairs:
        preprocess_pair(src, tgt, s_code, t_code, tokenizer)

if __name__ == "__main__":
    main()
