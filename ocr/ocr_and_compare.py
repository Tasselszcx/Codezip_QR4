"""
简单的多页 OCR + 合并 + 对比脚本
用法: python ocr_and_compare.py <图片目录> <code_id>
"""
import os
import sys
import json
import glob
import base64
import time
from openai import OpenAI


def load_api_key():
    """加载 API Key"""
    api_key = os.getenv("AIHUBMIX_API_KEY")
    if not api_key:
        # 尝试从多个位置查找 .env 文件
        script_dir = os.path.dirname(os.path.abspath(__file__))
        env_paths = [
            ".env",  # 当前目录
            os.path.join(script_dir, ".env"),  # 脚本所在目录
            os.path.join(script_dir, "..", ".env"),  # 上级目录
        ]
        
        for env_file in env_paths:
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.strip().startswith("AIHUBMIX_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
                if api_key:
                    break
    return api_key


def ocr_image(image_path, api_key, max_retries=3):
    """对单张图片进行 OCR，带重试"""
    print(f"  [OCR] {os.path.basename(image_path)}", end='', flush=True)
    
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"
    
    client = OpenAI(api_key=api_key, base_url="https://aihubmix.com/v1")
    
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model="glm-4.6v",
                temperature=0.0,
                max_tokens=4096,
                messages=[
                    {"role": "system", "content": "You are an OCR engine for code images."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the code exactly. Output plain text only. Preserve all whitespace and indentation."},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    },
                ],
            )
            
            text = (resp.choices[0].message.content or "").strip()
            text = text.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
            
            if text:
                print(f" -> {len(text)} chars")
                return text
            else:
                print(f" -> empty (retry {attempt+1})", flush=True)
                if attempt < max_retries - 1:
                    time.sleep(2)
        except Exception as e:
            print(f" -> error: {e} (retry {attempt+1})", flush=True)
            if attempt < max_retries - 1:
                time.sleep(2)
    
    print(f" -> FAILED")
    return ""


def calculate_metrics(reference, hypothesis):
    """计算 CER, WER, BLEU"""
    def lev_dist(s1, s2):
        if len(s1) < len(s2):
            return lev_dist(s2, s1)
        if len(s2) == 0:
            return len(s1)
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        return previous_row[-1]
    
    # CER
    cer = lev_dist(reference, hypothesis) / len(reference) * 100
    
    # WER
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    wer = lev_dist(ref_words, hyp_words) / len(ref_words) * 100
    
    # BLEU
    from collections import Counter
    import math
    
    precisions = []
    for n in range(1, 5):
        ref_ngrams = Counter([tuple(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)])
        hyp_ngrams = Counter([tuple(hyp_words[i:i+n]) for i in range(len(hyp_words)-n+1)])
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        precisions.append(matches / total if total > 0 else 0)
    
    if any(p == 0 for p in precisions):
        bleu = 0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        bp = 1.0 if len(hyp_words) >= len(ref_words) else math.exp(1 - len(ref_words) / len(hyp_words))
        bleu = bp * geo_mean * 100
    
    return cer, wer, bleu


def main():
    # 示例路径
    example_dir = r"D:\llm_projects\CodeZip\experiment_output\images\crypto-trader-bot-with-AI-algo_indicator_calculator.py\1024x1024_hl_nl"
    example_code_id = "crypto-trader-bot-with-AI-algo_indicator_calculator.py"
    
    # 获取参数（支持命令行或交互式输入）
    if len(sys.argv) >= 3:
        image_dir = sys.argv[1]
        code_id = sys.argv[2]
    else:
        print("Usage: python ocr_and_compare.py <image_dir> <code_id>")
        print("\n" + "=" * 60)
        print("Interactive Mode")
        print("=" * 60)
        
        # 显示示例
        print(f"\nExample directory: {example_dir}")
        print(f"Example code_id: {example_code_id}")
        
        # 询问是否使用示例
        use_example = input("\nUse example path? (y/n, default=y): ").strip().lower()
        
        if use_example == '' or use_example == 'y':
            image_dir = example_dir
            code_id = example_code_id
            print(f"Using: {image_dir}")
            print(f"Code ID: {code_id}")
        else:
            # 交互式输入
            image_dir = input("\nEnter image directory: ").strip()
            code_id = input("Enter code_id: ").strip()
            
            if not image_dir or not code_id:
                print("[ERROR] Both parameters are required")
                sys.exit(1)
    
    # Load API Key
    api_key = load_api_key()
    if not api_key:
        print("[ERROR] AIHUBMIX_API_KEY not found")
        sys.exit(1)
    
    # Get ratio
    print("\nEnter compression ratio (1, 2, 4, 8): ", end='')
    ratio = input().strip()
    
    # Find images (ratio=1 means original images without ratio suffix)
    if ratio == '1':
        pattern = os.path.join(image_dir, "page_*.png")
        # 排除带 ratio 后缀的文件
        all_images = glob.glob(pattern)
        images = sorted([img for img in all_images if not any(f'_ratio{r}.png' in img for r in [2, 4, 8])])
    else:
        pattern = os.path.join(image_dir, f"page_*_ratio{ratio}.png")
        images = sorted(glob.glob(pattern))
    
    if not images:
        print(f"[ERROR] No images found: {pattern}")
        sys.exit(1)
    
    print(f"\nFound {len(images)} images")
    
    # OCR each image
    print(f"\nStarting OCR...")
    ocr_results = []
    for img in images:
        text = ocr_image(img, api_key, max_retries=3)
        ocr_results.append(text)
    
    # Merge
    merged_ocr = '\n'.join(ocr_results)
    print(f"\nMerged: {len(merged_ocr)} chars, {len(merged_ocr.splitlines())} lines")
    
    # Load original code - 使用绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(script_dir, "..", "experiment_output", "dataset.json")
    
    if not os.path.exists(dataset_path):
        print(f"[ERROR] dataset.json not found at: {dataset_path}")
        sys.exit(1)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = {item['id']: item for item in json.load(f)}
    
    if code_id not in dataset:
        print(f"[ERROR] code_id not found: {code_id}")
        sys.exit(1)
    
    original_code = dataset[code_id]['code']
    print(f"Original: {len(original_code)} chars, {len(original_code.splitlines())} lines")
    
    # Calculate metrics
    print(f"\nCalculating metrics...")
    cer, wer, bleu = calculate_metrics(original_code, merged_ocr)
    
    # Output
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"CER (Character Error Rate):  {cer:.2f}%")
    print(f"WER (Word Error Rate):       {wer:.2f}%")
    print(f"BLEU Score:                  {bleu:.2f}")
    
    # Line matching
    ref_lines = original_code.splitlines()
    ocr_lines = merged_ocr.splitlines()
    matches = sum(1 for i in range(min(len(ref_lines), len(ocr_lines))) 
                  if ref_lines[i].strip() == ocr_lines[i].strip())
    emr = matches / max(len(ref_lines), len(ocr_lines)) * 100
    print(f"Exact Match Rate:            {emr:.2f}% ({matches}/{max(len(ref_lines), len(ocr_lines))} lines)")
    
    print("=" * 60)
    
    # Save
    output_file = f"ocr_merged_ratio{ratio}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(merged_ocr)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    main()
