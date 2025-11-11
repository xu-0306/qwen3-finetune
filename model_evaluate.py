# 匯入必要的函式庫
import evaluate  # Hugging Face Evaluate 函式庫
import nltk  # 仍然可以用於其他目的，或者如果您想比較
import os  # 用於處理檔案路徑


# ！！！重要提示！！！
# 以下檢查確保此腳本檔案的名稱不是 'evaluate.py'，以避免與 Hugging Face 的 'evaluate' 函式庫衝突。
# 如果您的腳本檔案名稱正好是 'evaluate.py'，它會與 Hugging Face 的 'evaluate' 函式庫衝突，
# 可能導致 "AttributeError: module 'evaluate' has no attribute 'load'" 錯誤。

def load_your_model_and_tokenizer(model_path):
    """
    載入您微調後的模型和分詞器。
    這部分需要您根據您使用的模型框架 (例如 Hugging Face Transformers) 進行實作。
    """
    print(f"模擬：應在此處實作模型載入邏輯: {model_path}")
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # print(f"正在從 {model_path} 載入模型和分詞器...")
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path)
    # print(f"模型和分詞器已從 {model_path} 載入。")
    # return model, tokenizer
    return None, None


def generate_translation(model, tokenizer, source_text):
    """
    使用您載入的模型和分詞器來產生翻譯。
    這部分也需要您根據您的模型進行實作。
    """
    if model is None or tokenizer is None:
        if "你好" in source_text:
            return "hello world"
        elif "天氣" in source_text:
            return "the weather is great today"
        else:
            return "i love python programming"

    # 模擬真實模型的翻譯邏輯
    # print(f"模擬：應在此處實作翻譯產生邏輯，輸入文本: {source_text}")
    # inputs = tokenizer(source_text, return_tensors="pt")
    # outputs = model.generate(**inputs)
    # translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return translated_text
    return "this is a simulated translation from your model"


if __name__ == "__main__":
    # --- 0. 檢查腳本檔案名稱 ---
    script_filename = os.path.basename(__file__)
    if script_filename == "evaluate.py":
        print(f"錯誤：此腳本檔案 ({__file__}) 的名稱正好是 'evaluate.py'，這會與 Hugging Face 的 'evaluate' 函式庫衝突。")
        print("請將此檔案重新命名 (例如 'run_my_evaluation.py' 或 'translation_eval_script.py') 然後再試一次。")
        exit()
    else:
        print(f"腳本檔案名稱 '{script_filename}' 已通過檢查。")

    # --- 1. 準備您的資料 ---
    source_sentences = [
        "你好，世界！",
        "今天天氣真好。",
        "我喜歡用 Python 編程。"
    ]
    reference_translations_corpus_hf = [
        ["hello, world!", "hi, world!"],
        ["the weather is nice today.", "it is a fine day today."],
        ["i like to program in python.", "i enjoy programming with python."]
    ]

    # --- 2. 載入您的模型 ---
    model_path = "G:/_python/Qwen3 1.7B finetune/merged_qwen3_1_7b_with_lora"
    model, tokenizer = load_your_model_and_tokenizer(model_path)

    # --- 3. 載入 BLEU 指標 ---
    bleu_metric = None
    print("\n--- 正在載入 BLEU 指標 ---")
    try:
        print("嘗試呼叫 evaluate.load('bleu')...")
        loaded_object = evaluate.load("bleu")

        print(f"偵錯：evaluate.load('bleu') 返回的物件: {loaded_object}")
        print(f"偵錯：該物件的類型: {type(loaded_object)}")

        if loaded_object is not None and hasattr(loaded_object, 'compute'):
            bleu_metric = loaded_object
            print("Hugging Face Evaluate 的 BLEU 指標已成功載入。")
        else:
            print("錯誤：evaluate.load('bleu') 返回了 None 或無效的物件 (沒有 'compute' 方法)。無法載入 BLEU 指標。")
            print(f"返回的物件: {loaded_object}")
            print("這可能是因為 'sacrebleu' 函式庫未正確安裝或初始化，或者網路問題導致無法下載必要元件。")
            print(f"請確認您已在環境中正確安裝 'evaluate' 和 'sacrebleu'。")
            print("如果問題持續，請檢查您的網路連線，或嘗試清除 Hugging Face 快取 (通常在 ~/.cache/huggingface/evaluate 和 ~/.cache/huggingface/modules)。")

    except AttributeError as e:
        if "'evaluate' has no attribute 'load'" in str(e):
            print(f"載入 BLEU 指標失敗 (AttributeError): {e}")
            print("這通常是因為您的腳本檔案名稱與 'evaluate' 函式庫衝突。")
            print("請確保您的腳本檔案名稱不是 'evaluate.py'。")
        else:
            print(f"載入 BLEU 指標時發生未預期的 AttributeError: {e}")
        print("另外，請確保您已安裝 'evaluate' 和 'sacrebleu' 函式庫。")
    except Exception as e:
        print(f"載入 BLEU 指標時發生錯誤: {e}")
        print("請確保您已安裝 'evaluate' 和 'sacrebleu' 函式庫，並檢查您的網路連線。")

    # --- 4. 產生翻譯並收集 ---
    candidate_translations_hf = []
    if model is None or tokenizer is None:
        print("\n警告：模型和分詞器未載入 (load_your_model_and_tokenizer 返回 None)。將使用模擬翻譯進行演示。")
        print("請實作 `load_your_model_and_tokenizer` 和 `generate_translation` 函數以使用您的實際模型。")

    print("\n--- 開始產生翻譯 ---")
    for i in range(len(source_sentences)):
        source_text = source_sentences[i]
        candidate_text = generate_translation(model, tokenizer, source_text)
        candidate_translations_hf.append(candidate_text)

        print(f"\n原始文本: {source_text}")
        print(f"模型翻譯: {candidate_text}")
        print(f"參考翻譯: {reference_translations_corpus_hf[i]}")

    # --- 5. 計算 BLEU 分數 ---
    print(f"\n--- 準備計算 BLEU 分數 ---")
    print(f"偵錯：計算前 bleu_metric 的值: {bleu_metric}")
    print(f"偵錯：計算前 bleu_metric 的類型: {type(bleu_metric)}")
    print(f"偵錯：bool(bleu_metric) 的結果: {bool(bleu_metric)}")
    print(f"偵錯：bleu_metric is not None: {bleu_metric is not None}")
    print(f"偵錯：candidate_translations_hf: {candidate_translations_hf}")
    print(f"偵錯：bool(candidate_translations_hf): {bool(candidate_translations_hf)}")
    print(f"偵錯：reference_translations_corpus_hf: {reference_translations_corpus_hf}")
    print(f"偵錯：bool(reference_translations_corpus_hf): {bool(reference_translations_corpus_hf)}")

    # *** 修改後的條件判斷 ***
    condition_bleu_loaded = bleu_metric is not None
    condition_candidates_exist = bool(candidate_translations_hf)
    condition_references_exist = bool(reference_translations_corpus_hf)

    print(f"偵錯：條件 bleu_metric is not None: {condition_bleu_loaded}")
    print(f"偵錯：條件 candidate_translations_hf 非空: {condition_candidates_exist}")
    print(f"偵錯：條件 reference_translations_corpus_hf 非空: {condition_references_exist}")

    if condition_bleu_loaded and condition_candidates_exist and condition_references_exist:
        if len(candidate_translations_hf) == len(reference_translations_corpus_hf):
            print("\n正在計算 BLEU 分數...")
            try:
                results = bleu_metric.compute(predictions=candidate_translations_hf,
                                              references=reference_translations_corpus_hf)
                print("\n--- Hugging Face Evaluate BLEU 結果 ---")
                bleu_score_val = results.get('bleu')
                if bleu_score_val is not None:
                    print(f"BLEU 分數: {bleu_score_val:.4f}")
                else:
                    print(f"BLEU 分數: N/A (未在結果中找到 'bleu' 鍵)")

                print(f"各 n-gram 的準確率 (Precisions): {results.get('precisions', 'N/A')}")

                brevity_penalty_val = results.get('brevity_penalty')
                if brevity_penalty_val is not None:
                    print(f"簡潔懲罰 (Brevity Penalty): {brevity_penalty_val:.4f}")
                else:
                    print(f"簡潔懲罰 (Brevity Penalty): N/A")

                print(f"翻譯長度 (Translation Length): {results.get('translation_length', 'N/A')}")
                print(f"參考長度 (Reference Length): {results.get('reference_length', 'N/A')}")
            except Exception as e:
                print(f"計算 BLEU 分數時發生錯誤: {e}")
                print("可能的原因：")
                print("- 檢查 'predictions' 和 'references' 的格式是否符合 'evaluate.load(\"bleu\")' 的要求。")
                print("  'predictions' 應該是字串列表。")
                print("  'references' 應該是列表的列表，其中每個內部列表包含一個或多個參考翻譯字串。")
                print(f"  目前的 predictions: {candidate_translations_hf}")
                print(f"  目前的 references: {reference_translations_corpus_hf}")
        else:
            print("錯誤：候選翻譯的數量與參考翻譯的數量不匹配。")
            print(f"候選翻譯數量: {len(candidate_translations_hf)}")
            print(f"參考翻譯數量: {len(reference_translations_corpus_hf)}")

    # 修改後的 elif 結構，使其更清晰
    elif not condition_bleu_loaded: # 即 bleu_metric is None
        print("\nBLEU 指標未成功載入 (bleu_metric is None)。無法計算分數。請檢查先前的錯誤和偵錯訊息。")
    elif not condition_candidates_exist:
        print("\n候選翻譯列表為空 (candidate_translations_hf is empty)。無法計算 BLEU 分數。")
    elif not condition_references_exist:
        print("\n參考翻譯列表為空 (reference_translations_corpus_hf is empty)。無法計算 BLEU 分數。")
    else:
        # 這種情況理論上不應該發生，因為上面的條件已經覆蓋了主要情況
        print("\n未知原因導致無法計算 BLEU 分數，儘管所有單獨條件似乎都滿足。請檢查偵錯輸出。")

    print("\n腳本執行完畢。")