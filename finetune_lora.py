# =============================================================
#  finetune_lora.py  (v1.3)
#  ──────────────────────────────────────────────────────────────
#  1. 監控 qwen3_cje_lora/tokenized_* 目錄，資料備妥後自動啟動微調。
# 2. 與舊版 Transformers 相容：僅在 TrainingArguments 支援時
#    傳入 evaluation_strategy / eval_steps / load_best_model_at_end
# =============================================================

from __future__ import annotations
import os, argparse, warnings, time, inspect
from pathlib import Path

import torch
from datasets import load_from_disk, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
    get_peft_model,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

warnings.filterwarnings("ignore", category=UserWarning)

# -------------------------------------------------------------
#  GLOBAL CONFIG
# -------------------------------------------------------------
MODEL_NAME_OR_PATH = os.getenv(
    "QWEN_MODEL",
    "/mnt/g/_python/Qwen3 1.7B finetune/_Models/Qwen3-1.7B",
)
OUTPUT_ROOT = Path("qwen3_cje_lora")
MAX_SEQ_LEN = 2048

# LoRA 超參數
LORA_R       = 32
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# -------------------------------------------------------------
#  HELPERS
# -------------------------------------------------------------

def ensure_tokenizer_special_tokens(tokenizer):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def wait_for_tokenized_dirs(poll_sec: int) -> list[Path]:
    """輪詢 OUTPUT_ROOT 下的 tokenized_* 目錄直到出現"""
    while True:
        dirs = list((OUTPUT_ROOT).glob("tokenized_*"))
        if dirs:
            print(f"✔ 偵測到 {len(dirs)} 個 tokenized_* 目錄，開始微調…")
            return dirs
        print(f"[waiting] 尚未找到 tokenized_* 目錄，{poll_sec}s 後重試…")
        time.sleep(poll_sec)

# -------------------------------------------------------------
#  FINE‑TUNE FUNCTION
# -------------------------------------------------------------

def finetune(subset: float, epochs: int, poll_sec: int):
    # ─── 1. 準備 tokenizer & model ───────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_ROOT / "tokenizer", trust_remote_code=True)
    ensure_tokenizer_special_tokens(tokenizer)

    use_bf16 = torch.cuda.is_bf16_supported()
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        device_map="auto",
        quantization_config=bnb_cfg,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    ))

    # ─── 2. 等待並讀取資料 ──────────────────────────────────────
    token_dirs = wait_for_tokenized_dirs(poll_sec)
    data = concatenate_datasets([load_from_disk(p) for p in token_dirs])

    if subset < 1.0:
        size = max(int(len(data) * subset), 1)
        data = data.select(range(size))
        print(f"Sub‑sampling {subset*100:.1f}% → {len(data):,} samples")

    split = data.train_test_split(test_size=0.05, seed=42)

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
    )

    # ─── 3. 建立 TrainingArguments (兼容舊版) ──────────────────
    eff_bs, per_dev_bs = 64, 8
    grad_acc = eff_bs // per_dev_bs

    base_kwargs = dict(
        output_dir=str(OUTPUT_ROOT / "lora_ckpt"),
        per_device_train_batch_size=per_dev_bs,
        gradient_accumulation_steps=grad_acc,
        learning_rate=2e-4,
        num_train_epochs=epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        fp16=not use_bf16,
        bf16=use_bf16,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        group_by_length=True,
        report_to=[],
    )

    sig = inspect.signature(TrainingArguments.__init__)
    has_eval = "evaluation_strategy" in sig.parameters
    has_save = "save_strategy" in sig.parameters

    if has_eval:
        base_kwargs["evaluation_strategy"] = "steps"
    if has_save:
        base_kwargs["save_strategy"] = "steps"

    # 只有在同時支援 evaluation_strategy 時才加入最佳模型相關設定
    if has_eval and "load_best_model_at_end" in sig.parameters:
        if "eval_steps" in sig.parameters:
            base_kwargs["eval_steps"] = 200
        base_kwargs["load_best_model_at_end"] = True
        if "metric_for_best_model" in sig.parameters:
            base_kwargs["metric_for_best_model"] = "eval_loss"

    train_args = TrainingArguments(**base_kwargs)

        # ─── 4. SFTTrainer ──────────────────────────────────────────
    cand_kwargs = {
        "model": model,
        "args": train_args,
        "train_dataset": split["train"],
        "eval_dataset": split["test"],
        "data_collator": collator,
        "max_seq_length": MAX_SEQ_LEN,
        "dataset_text_field": None,
    }
    sig_sft = inspect.signature(SFTTrainer.__init__).parameters
    sft_kwargs = {k: v for k, v in cand_kwargs.items() if k in sig_sft}

    trainer = SFTTrainer(**sft_kwargs)

    # ─── 5.  開始訓練 ────────────────────────────────────────────
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_ROOT / "lora_ckpt" / "tokenizer")

# -------------------------------------------------------------
#  CLI ENTRY
# -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=float, default=1.0, help="使用資料比例 0‑1")
    parser.add_argument("--epochs", type=int, default=3, help="訓練 epoch 數")
    parser.add_argument("--poll_sec", type=int, default=0, help="輪詢等待秒數")
    args = parser.parse_args()
    finetune(args.subset, args.epochs, args.poll_sec)

if __name__ == "__main__":
    main()
