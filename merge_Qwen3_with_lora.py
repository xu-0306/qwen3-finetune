import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 配置路徑
base_model_path = r"G:\_python\Qwen3 1.7B finetune\_Models\Qwen3-1.7B"
adapter_path = r"./qwen3_1_7b_cje_translator_finetuned_qlora_batched_v8_dclm_subset/final_qlora_adapters" # 替換 X
output_merged_model_path = r"./merged_qwen3_1_7b_with_lora" # 您希望保存合併後模型的路徑

# 2. 加載分詞器 (從適配器文件夾加載)
print(f"從 '{adapter_path}' 加載分詞器...")
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
# (處理 pad_token 的 fallback 邏輯，如果需要)
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    # ... (其他 fallback)

# 3. 加載基礎模型 (建議加載原始精度，以便後續量化為 GGUF 時有更好的起點)
print(f"從 '{base_model_path}' 加載基礎模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16, # 或者 torch.bfloat16，取決於模型和硬體
    low_cpu_mem_usage=True,    # 嘗試減少 CPU 內存使用
    trust_remote_code=True
)

# 如果 tokenizer 被擴展過
if len(tokenizer) > base_model.get_input_embeddings().num_embeddings:
    base_model.resize_token_embeddings(len(tokenizer))

# 4. 加載 LoRA 適配器
print(f"從 '{adapter_path}' 加載 LoRA 適配器...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# 5. 合併 LoRA 權重到基礎模型中
print("正在合併 LoRA 適配器到基礎模型...")
model = model.merge_and_unload() # 關鍵步驟！
print("LoRA 適配器已合併並卸載。")

# 6. 保存合併後的完整模型和分詞器
print(f"正在保存合併後的模型到 '{output_merged_model_path}'...")
model.save_pretrained(output_merged_model_path)
tokenizer.save_pretrained(output_merged_model_path)

print(f"合併後的模型和分詞器已成功保存到 '{output_merged_model_path}'")