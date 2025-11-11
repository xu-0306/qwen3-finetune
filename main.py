import os
import torch
import random # 用於打亂數據集
from datasets import load_dataset, Dataset # 從 datasets 函式庫匯入 Dataset 以創建數據集對象
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)

# Suppress UserWarnings for a cleaner output (optional)
# 忽略使用者警告，讓輸出更簡潔
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main():
    # --------------------------------------------------------------------------
    # 1. Configuration (設定區)
    # --------------------------------------------------------------------------
    # **重要**: 請將此路徑修改為您本地存放 Qwen3-1.7B **非 FP8 版本**模型檔案的資料夾路徑
    model_name_or_path = r"G:\_python\Qwen3 1.7B finetune\_Models\Qwen3-1.7B"  # <--- 請務必修改此路徑為您的實際非 FP8 模型路徑

    lora_r = 16 # LoRA 的秩
    lora_alpha = 32 # LoRA 的 alpha 縮放因子
    lora_dropout = 0.05 # LoRA 層的 dropout 比率
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    output_dir = "./qwen3_1_7b_cje_translator_local_model_qlora" # 輸出目錄
    per_device_train_batch_size = 1 # 每個 GPU 的訓練批次大小
    gradient_accumulation_steps = 8 # 梯度累積步數
    learning_rate = 2e-4 # 學習率
    num_train_epochs = 3 # 訓練總輪數
    use_bf16 = torch.cuda.is_bf16_supported() # 檢查是否支援 bfloat16
    max_seq_length = 1024 # 輸入序列的最大長度

    # --------------------------------------------------------------------------
    # 2. Load Tokenizer (載入分詞器)
    # --------------------------------------------------------------------------
    print(f"從路徑載入分詞器: {model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        print(f"分詞器的 pad_token_id 未設定。將 pad_token_id 設為 eos_token_id: {tokenizer.eos_token_id}")
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer.pad_token_id == tokenizer.bos_token_id:
         print(f"分詞器的 pad_token_id 與 bos_token_id 相同。將 pad_token_id 設為 eos_token_id: {tokenizer.eos_token_id}")
         tokenizer.pad_token_id = tokenizer.eos_token_id
         tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"分詞器 pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    print(f"分詞器 eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    print(f"分詞器 bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")

    # --------------------------------------------------------------------------
    # 3. Load Model with QLoRA configuration (載入 QLoRA 設定的模型)
    # --------------------------------------------------------------------------
    print(f"從路徑載入模型: {model_name_or_path} 並使用 QLoRA (4-bit 量化)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    print("LoRA 應用後的可訓練參數:")
    model.print_trainable_parameters()

    # --------------------------------------------------------------------------
    # 4. Data Preparation (數據準備 - 使用 OpenSubtitles-TW-Corpus 的 Parquet 檔案)
    # --------------------------------------------------------------------------
    print("準備 OpenSubtitles-TW-Corpus 數據 (使用 Parquet 檔案)...")
    dataset_script_path = r"G:\_python\Qwen3 1.7B finetune\_dataset\OpenSubtitles-TW-Corpus"

    TARGET_CONFIGS_AND_LANGS = {
        "en-zh_tw": ("en", "zh_tw"),
        "ja-zh_tw": ("ja", "zh_tw"),
    }
    LANG_NAME_MAP = {
        "zh_tw": "Traditional Chinese",
        "en": "English",
        "ja": "Japanese",
    }
    all_processed_examples = []
    for config_name, (source_code, target_code) in TARGET_CONFIGS_AND_LANGS.items():
        source_lang_full = LANG_NAME_MAP.get(source_code)
        target_lang_full = LANG_NAME_MAP.get(target_code)
        if not source_lang_full or not target_lang_full:
            print(f"警告: 設定 '{config_name}' 中的語言代碼 '{source_code}' 或 '{target_code}' 未知，跳過。")
            continue
        print(f"載入數據集設定: {config_name} (處理 {source_lang_full} <-> {target_lang_full})")
        try:
            dataset = load_dataset(dataset_script_path, name=config_name, split="train")
        except Exception as e:
            print(f"載入數據集設定 {config_name} 時發生錯誤: {e}")
            continue
        file_example_count = 0
        for example in dataset:
            instruction = example.get("instruction", "").strip()
            input_text = example.get("input", "").strip()
            output_text = example.get("output", "").strip()
            if not input_text or not output_text: continue
            all_processed_examples.append({"instruction_prompt": instruction, "source_text_for_user": input_text, "target_text_for_assistant": output_text})
            file_example_count +=1
            reversed_instruction = f"Translate the following {target_lang_full} text to {source_lang_full}:"
            all_processed_examples.append({"instruction_prompt": reversed_instruction, "source_text_for_user": output_text, "target_text_for_assistant": input_text})
            file_example_count += 1
        print(f"從設定 '{config_name}' 添加了 {file_example_count} 個樣本。")
    if not all_processed_examples: raise ValueError("沒有載入任何數據。")
    random.shuffle(all_processed_examples)
    print(f"總共處理並打亂了 {len(all_processed_examples)} 個樣本。")
    num_total_examples = len(all_processed_examples)
    num_train_examples = int(num_total_examples * 0.95)
    train_data_list = all_processed_examples[:num_train_examples]
    eval_data_list = all_processed_examples[num_train_examples:]
    if not train_data_list: raise ValueError("分割後訓練數據列表為空。")
    if not eval_data_list and num_total_examples > 0 : print("警告: 分割後驗證數據列表為空。")
    train_dataset_hf = Dataset.from_list(train_data_list)
    eval_dataset_hf = Dataset.from_list(eval_data_list) if eval_data_list else None

    def format_data_for_qwen3(example):
        messages = [
            {"role": "system", "content": "You are a precise and helpful multilingual translation assistant for Chinese, Japanese, and English."},
            {"role": "user", "content": f"{example['instruction_prompt']}\n{example['source_text_for_user']}"},
            {"role": "assistant", "content": example['target_text_for_assistant']}
        ]
        formatted_text_with_response = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        tokenized_full = tokenizer(
            formatted_text_with_response, truncation=True, max_length=max_seq_length, padding=False,
        )
        labels = list(tokenized_full['input_ids'])
        prompt_messages_for_masking = messages[:-1]
        templated_prompt_for_masking = tokenizer.apply_chat_template(
            prompt_messages_for_masking, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        tokenized_prompt_for_masking = tokenizer(templated_prompt_for_masking, truncation=True, max_length=max_seq_length, padding=False)
        prompt_len = len(tokenized_prompt_for_masking['input_ids'])
        for i in range(min(prompt_len, len(labels))): labels[i] = -100
        tokenized_full['labels'] = labels
        return tokenized_full

    print("對數據集進行分詞和格式化...")
    # **重要**: 在 Windows 上，如果 .map() 使用多進程 (num_proc > 1) 出現 PermissionError，請將 num_proc 設為 1
    num_map_workers = 1 if os.name == 'nt' else max(1, os.cpu_count() // 2) # Windows 設為 1，其他系統使用多核心
    print(f"使用 {num_map_workers} 個進程進行 .map() 操作。")

    tokenized_train_dataset = train_dataset_hf.map(format_data_for_qwen3, remove_columns=train_dataset_hf.column_names, num_proc=num_map_workers)
    if eval_dataset_hf:
        tokenized_eval_dataset = eval_dataset_hf.map(format_data_for_qwen3, remove_columns=eval_dataset_hf.column_names, num_proc=num_map_workers)
        tokenized_eval_dataset = tokenized_eval_dataset.filter(lambda example: example.get('input_ids') is not None and example.get('labels') is not None)
    else:
        tokenized_eval_dataset = None
    tokenized_train_dataset = tokenized_train_dataset.filter(lambda example: example.get('input_ids') is not None and example.get('labels') is not None)
    if not tokenized_train_dataset: raise ValueError("訓練數據集在分詞後為空。")
    if eval_dataset_hf and not tokenized_eval_dataset: print("警告: 驗證數據集在分詞後為空。")
    print(f"分詞後的訓練樣本數量: {len(tokenized_train_dataset)}")
    if tokenized_eval_dataset: print(f"分詞後的驗證樣本數量: {len(tokenized_eval_dataset)}")
    if len(tokenized_train_dataset) > 0:
        print(f"抽樣分詞後輸入 (解碼後): {tokenizer.decode(tokenized_train_dataset[0]['input_ids'])}")
        print(f"抽樣標籤 (未遮罩部分解碼後): {tokenizer.decode([l for l in tokenized_train_dataset[0]['labels'] if l != -100])}")

    # --------------------------------------------------------------------------
    # 5. Training Arguments (設定訓練參數)
    # --------------------------------------------------------------------------
    print("定義訓練參數...")
    # **重要**: 在 Windows 上，dataloader_num_workers > 0 可能導致問題，建議設為 0
    dataloader_workers = 0 if os.name == 'nt' else max(1, os.cpu_count() // 4)
    print(f"使用 {dataloader_workers} 個 dataloader workers。")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(100, len(tokenized_train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps * 4)) if tokenized_train_dataset and len(tokenized_train_dataset) > 0 else 100,
        evaluation_strategy="steps" if tokenized_eval_dataset and len(tokenized_eval_dataset)>0 else "no",
        eval_steps=max(100, len(tokenized_train_dataset) // (per_device_train_batch_size * gradient_accumulation_steps * 4)) if tokenized_eval_dataset and len(tokenized_eval_dataset)>0 else None,
        load_best_model_at_end=True if tokenized_eval_dataset and len(tokenized_eval_dataset)>0 else False,
        metric_for_best_model="eval_loss" if tokenized_eval_dataset and len(tokenized_eval_dataset)>0 else None,
        optim="paged_adamw_8bit",
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        report_to="tensorboard",
        save_total_limit=2,
        dataloader_num_workers=dataloader_workers, # 修改此處
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --------------------------------------------------------------------------
    # 6. Initialize Trainer (初始化訓練器)
    # --------------------------------------------------------------------------
    print("初始化訓練器...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset if tokenized_eval_dataset and len(tokenized_eval_dataset)>0 else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --------------------------------------------------------------------------
    # 7. Start Fine-tuning (開始微調)
    # --------------------------------------------------------------------------
    print("開始微調...")
    try:
        trainer.train()
    except Exception as e:
        print(f"訓練過程中發生錯誤: {e}")
        error_checkpoint_path = os.path.join(output_dir, "error_checkpoint")
        if hasattr(trainer, 'model') and trainer.model is not None:
             trainer.model.save_pretrained(error_checkpoint_path)
             tokenizer.save_pretrained(error_checkpoint_path)
             print(f"因錯誤已儲存模型檢查點於: {error_checkpoint_path}")
        raise

    # --------------------------------------------------------------------------
    # 8. Save the fine-tuned LoRA adapters and tokenizer (儲存微調後的 LoRA 適配器和分詞器)
    # --------------------------------------------------------------------------
    final_lora_path = os.path.join(output_dir, "final_qlora_adapters")
    print(f"儲存最終的 LoRA 適配器于: {final_lora_path}")
    model.save_pretrained(final_lora_path)
    tokenizer.save_pretrained(final_lora_path)

    print("微調完成!")
    print(f"LoRA 適配器和分詞器已儲存至: {final_lora_path}")

if __name__ == "__main__":
    main()
