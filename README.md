# Qwen3 1.7B 多語言翻譯模型微調系統 - 使用教學文檔

## 📋 專案概述

本專案是基於 **Qwen3-1.7B** 模型的多語言翻譯系統，使用 **QLoRA** (量化 LoRA) 技術進行高效微調，支援**繁體中文、英文、日文**之間的雙向翻譯。

---

## 🔧 核心程式功能說明

### 1. `Preprocess_Data_Moses.py` - 資料預處理

**功能**：將 Moses 格式的平行語料轉換為模型訓練所需的 tokenized 格式

**主要功能**：

- 掃描 `_dataset/` 目錄，自動發現平行語料對（如 corpus.en + corpus.zh_tw）
- 自動配對不同語言的檔案
- 生成雙向翻譯樣本（A→B 和 B→A）
- 添加語言標籤（`<en>` `<zh_TW>` `<ja>`）
- 使用 Qwen3 tokenizer 進行 tokenization
- 保存為 Arrow 格式到 `qwen3_cje_lora/tokenized_*/`

**支援的語言擴展名**：`.en` (英文)、`.ja` (日文)、`.zh` / `.zh_tw` / `.zh_hant` (繁體中文)

**使用方式**：

```bash
python Preprocess_Data_Moses.py
# 或指定資料目錄
python Preprocess_Data_Moses.py --data_root /path/to/data
```

---

### 2. `finetune_lora.py` - LoRA 微調訓練

**功能**：使用預處理的資料訓練 LoRA 適配器

**主要功能**：

- 加載 Qwen3-1.7B 基礎模型（4-bit 量化）
- 配置 LoRA 參數（r=32, alpha=64, dropout=0.05）
- 監控 `qwen3_cje_lora/tokenized_*` 目錄，資料準備好後自動啟動訓練
- 使用 QLoRA 技術進行參數高效微調
- 定期保存檢查點
- 支援驗證集評估

**訓練配置**：

- Batch size: 64（有效）= 8（每裝置）× 8（梯度累積）
- Learning rate: 2e-4
- Optimizer: paged_adamw_8bit
- 預設訓練 3 個 epoch

**使用方式**：

```bash
python finetune_lora.py
# 使用部分資料訓練（如 10%）
python finetune_lora.py --subset 0.1 --epochs 5
```

---

### 3. `merge_Qwen3_with_lora.py` - 模型合併

**功能**：將訓練好的 LoRA 適配器合併回基礎模型

**主要功能**：

- 加載 Qwen3-1.7B 基礎模型
- 加載訓練好的 LoRA 適配器
- 執行權重合併（`merge_and_unload()`）
- 保存完整的微調模型和 tokenizer
- 輸出可直接用於推理的完整模型

**使用方式**：

```bash
python merge_Qwen3_with_lora.py
```

**注意**：需在腳本中配置正確的路徑：

- `base_model_path`：基礎模型路徑
- `adapter_path`：LoRA 適配器路徑
- `output_merged_model_path`：輸出路徑

---

### 4. `model_evaluate.py` - 模型評估

**功能**：使用 BLEU 指標評估翻譯質量

**主要功能**：

- 加載合併後的模型
- 準備測試集（源語言句子 + 參考翻譯）
- 生成模型翻譯
- 計算 BLEU 分數和相關指標
- 顯示詳細的評估結果

**評估指標**：

- BLEU 分數
- 各 n-gram 準確率
- 簡潔懲罰（Brevity Penalty）
- 翻譯長度統計

**使用方式**：

```bash
python model_evaluate.py
```

### 6. `main.py` - 主程式

**功能**：整合推理功能的主程式

**主要功能**：

- 載入訓練好的模型
- 提供翻譯介面
- 支援互動式翻譯

---

## 🚀 基本使用流程

```
步驟 1: 準備資料 → 將平行語料放入 _dataset/ 目錄

步驟 2: 資料預處理 → python Preprocess_Data_Moses.py

步驟 3: LoRA 微調 → python finetune_lora.py

步驟 4: 合併模型 → python merge_Qwen3_with_lora.py

步驟 5: 評估模型 → python model_evaluate.py
```

---

## 💡 模型使用範例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 載入模型
model_path = "./merged_qwen3_1_7b_with_lora"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    trust_remote_code=True
)

# 翻譯函數
def translate(source_lang, target_lang, text):
    prompt = f"""<|im_start|>system
You are a precise and helpful multilingual translation assistant.<|im_end|>
<|im_start|>user
<{source_lang}> <{target_lang}>
{text}<|im_end|>
<|im_start|>assistant
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("<|im_start|>assistant")[-1].strip()

# 使用範例
print(translate("zh_tw", "en", "今天天氣真好。"))
print(translate("en", "ja", "Hello, how are you?"))
```

---

## 📚 參考資源

### 官方文檔

- **Qwen 官方文檔**: https://github.com/QwenLM/Qwen
- **Hugging Face Transformers**: https://huggingface.co/docs/transformers
- **PEFT 函式庫**: https://github.com/huggingface/peft

### 相關論文

- **LoRA 論文**: https://arxiv.org/abs/2106.09685
- **QLoRA 論文**: https://arxiv.org/abs/2305.14314

### 工具與函式庫

- **llama.cpp**: https://github.com/ggerganov/llama.cpp （用於 GGUF 格式轉換）
- **Datasets 函式庫**: https://huggingface.co/docs/datasets
- **TRL 函式庫**: https://github.com/huggingface/trl

---
