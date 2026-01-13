"""
详细分析 OCR 结果与原始代码的差异，计算各项指标

使用方法:
    python analyze_ocr_metrics.py <code_id> <compression_ratio>
    
示例:
    python analyze_ocr_metrics.py crypto-trader-bot-with-AI-algo_indicator_calculator.py 2
"""
import json
import sys
import os
import glob
from pathlib import Path


# 简单的编辑距离实现
def levenshtein_distance(s1, s2):
    """计算两个序列的编辑距离"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
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


# 简单的 BLEU 实现
def simple_bleu(reference, hypothesis, max_n=4):
    """计算 BLEU score (1-4 gram)"""
    from collections import Counter
    import math
    
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # 如果假设为空，返回0
    if not hyp_words:
        return 0.0
    
    # 计算 n-gram precision
    precisions = []
    for n in range(1, max_n + 1):
        ref_ngrams = Counter([tuple(ref_words[i:i+n]) for i in range(len(ref_words)-n+1)])
        hyp_ngrams = Counter([tuple(hyp_words[i:i+n]) for i in range(len(hyp_words)-n+1)])
        
        matches = sum((ref_ngrams & hyp_ngrams).values())
        total = sum(hyp_ngrams.values())
        
        if total == 0:
            precision = 0
        else:
            precision = matches / total
        
        precisions.append(precision)
    
    # 几何平均
    if any(p == 0 for p in precisions):
        geo_mean = 0
    else:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    
    # Brevity penalty
    bp = 1.0 if len(hyp_words) >= len(ref_words) else math.exp(1 - len(ref_words) / len(hyp_words))
    
    return bp * geo_mean


def load_dataset(dataset_path: str = "../experiment_output/dataset.json") -> dict:
    """从 dataset.json 加载原始代码数据"""
    if not os.path.exists(dataset_path):
        # 尝试当前目录
        dataset_path = "experiment_output/dataset.json"
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"找不到 dataset.json: {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        return {item['id']: item for item in json.load(f)}


def load_ocr_results(code_id: str, ratio: int, 
                     base_path: str = "../experiment_output") -> str:
    """
    加载指定 code_id 和 compression ratio 的所有 OCR 结果页面，并拼接
    
    Args:
        code_id: 代码文件 ID (如 "crypto-trader-bot-with-AI-algo_indicator_calculator.py")
        ratio: 压缩比例 (1, 2, 4, 8)
        base_path: 实验输出根目录
    
    Returns:
        拼接后的完整 OCR 文本
    """
    if not os.path.exists(base_path):
        base_path = "experiment_output"
    
    # 构建图片目录路径
    # 格式: experiment_output/images/{code_id}/1024x1024_hl_nl/
    image_dir = os.path.join(base_path, "images", code_id, "1024x1024_hl_nl")
    
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"找不到图片目录: {image_dir}")
    
    # 查找所有该 ratio 的 OCR 结果文件
    # 格式: page_001_ratio2_ocr.txt, page_002_ratio2_ocr.txt, ...
    pattern = os.path.join(image_dir, f"page_*_ratio{ratio}_ocr.txt")
    ocr_files = sorted(glob.glob(pattern))
    
    if not ocr_files:
        raise FileNotFoundError(f"找不到 OCR 结果文件: {pattern}")
    
    print(f"📄 找到 {len(ocr_files)} 个 OCR 结果文件:")
    for f in ocr_files:
        print(f"   - {os.path.basename(f)}")
    
    # 读取并拼接所有页面
    ocr_pages = []
    for ocr_file in ocr_files:
        with open(ocr_file, 'r', encoding='utf-8') as f:
            content = f.read()
            # 去除特殊标记
            content = content.replace('<|begin_of_box|>', '').replace('<|end_of_box|>', '')
            ocr_pages.append(content.strip())
    
    # 用换行符拼接所有页面
    full_ocr_text = '\n'.join(ocr_pages)
    
    print(f"✅ 拼接完成: {len(ocr_pages)} 页 → {len(full_ocr_text)} 字符\n")
    
    return full_ocr_text


def main():
    # 解析命令行参数
    if len(sys.argv) < 3:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <code_id> <compression_ratio>")
        print("\n示例:")
        print(f"  python {sys.argv[0]} crypto-trader-bot-with-AI-algo_indicator_calculator.py 2")
        print(f"  python {sys.argv[0]} moon-dev-ai-agents_src_config.py 1")
        print("\n可用的 code_id 请查看 experiment_output/dataset.json")
        sys.exit(1)
    
    code_id = sys.argv[1]
    ratio = int(sys.argv[2])
    
    print("=" * 80)
    print("📊 OCR 结果详细分析工具")
    print("=" * 80)
    print(f"Code ID: {code_id}")
    print(f"Compression Ratio: {ratio}x")
    print("=" * 80)
    print()
    
    # 1. 加载数据集
    print("1️⃣ 加载数据集...")
    try:
        dataset = load_dataset()
        print(f"✅ 数据集加载成功，包含 {len(dataset)} 个代码样本\n")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        sys.exit(1)
    
    # 2. 获取原始代码
    if code_id not in dataset:
        print(f"❌ 错误: 数据集中不存在 code_id '{code_id}'")
        print(f"\n可用的 code_id:")
        for cid in sorted(dataset.keys()):
            print(f"  - {cid}")
        sys.exit(1)
    
    original_code = dataset[code_id]['code']
    print(f"2️⃣ 原始代码:")
    print(f"   - 字符数: {len(original_code)}")
    print(f"   - 行数: {len(original_code.splitlines())}")
    print(f"   - 来源: {dataset[code_id]['repo']}")
    print()
    
    # 3. 加载并拼接 OCR 结果
    print(f"3️⃣ 加载 OCR 结果 (ratio={ratio})...")
    try:
        ocr_text = load_ocr_results(code_id, ratio)
    except Exception as e:
        print(f"❌ 加载 OCR 结果失败: {e}")
        sys.exit(1)
    
    # 4. 基本信息对比
    print("=" * 80)
    print("4️⃣ 基本信息对比")
    print("=" * 80)
    print(f"原始代码字符数: {len(original_code)}")
    print(f"原始代码行数: {len(original_code.splitlines())}")
    print(f"OCR 结果字符数: {len(ocr_text)}")
    print(f"OCR 结果行数: {len(ocr_text.splitlines())}")
    print(f"字符差异: {abs(len(original_code) - len(ocr_text))} ({abs(len(original_code) - len(ocr_text)) / len(original_code) * 100:.1f}%)")
    print(f"行数差异: {abs(len(original_code.splitlines()) - len(ocr_text.splitlines()))}")

    # 5. 计算评估指标
    print("\n" + "=" * 80)
    print("5️⃣ 计算评估指标")
    print("=" * 80)

    # === CER (Character Error Rate) ===
    def calculate_cer(reference, hypothesis):
        lev_dist = levenshtein_distance(reference, hypothesis)
        cer = lev_dist / len(reference) * 100
        return cer, lev_dist

    cer, lev_dist = calculate_cer(original_code, ocr_text)

    print(f"\n【CER - 字符错误率】")
    print(f"  {cer:.2f}% (编辑距离: {lev_dist})")

    # === WER (Word Error Rate) ===
    def calculate_wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        lev_dist = levenshtein_distance(ref_words, hyp_words)
        wer = lev_dist / len(ref_words) * 100
        return wer, lev_dist

    wer, wer_lev = calculate_wer(original_code, ocr_text)

    print(f"\n【WER - 单词错误率】")
    print(f"  {wer:.2f}% (编辑距离: {wer_lev})")

    # === BLEU Score ===
    def calculate_bleu(reference, hypothesis):
        return simple_bleu(reference, hypothesis)

    bleu = calculate_bleu(original_code, ocr_text)

    print(f"\n【BLEU Score】")
    print(f"  {bleu * 100:.2f}")

    # === Exact Match Rate (逐行比较) ===
    def calculate_exact_match_rate(reference, hypothesis):
        ref_lines = reference.splitlines()
        hyp_lines = hypothesis.splitlines()
        
        max_len = max(len(ref_lines), len(hyp_lines))
        matches = 0
        
        for i in range(max_len):
            ref_line = ref_lines[i] if i < len(ref_lines) else ""
            hyp_line = hyp_lines[i] if i < len(hyp_lines) else ""
            if ref_line.strip() == hyp_line.strip():
                matches += 1
        
        rate = matches / max_len * 100
        return rate, matches, max_len

    emr, match_count, total_lines = calculate_exact_match_rate(original_code, ocr_text)

    print(f"\n【Exact Match Rate - 逐行精确匹配率】")
    print(f"  {emr:.2f}% ({match_count}/{total_lines} 行匹配)")

    # 6. 逐行差异分析
    print("\n" + "=" * 80)
    print("6️⃣ 逐行差异分析 (前 30 行)")
    print("=" * 80)

    ref_lines = original_code.splitlines()
    ocr_lines = ocr_text.splitlines()

    max_display = min(30, max(len(ref_lines), len(ocr_lines)))
    diff_count = 0

    for i in range(max_display):
        ref_line = ref_lines[i] if i < len(ref_lines) else ""
        ocr_line = ocr_lines[i] if i < len(ocr_lines) else ""
        
        if ref_line.strip() == ocr_line.strip():
            status = "✅"
        else:
            status = "❌"
            diff_count += 1
        
        print(f"\n第 {i+1} 行 {status}")
        print(f"  原始: {repr(ref_line[:100])}")
        print(f"  OCR:  {repr(ocr_line[:100])}")

    if max(len(ref_lines), len(ocr_lines)) > max_display:
        remaining = max(len(ref_lines), len(ocr_lines)) - max_display
        print(f"\n... (还有 {remaining} 行未显示)")

    print(f"\n前 {max_display} 行中有 {diff_count} 行存在差异")

    # 7. 完整性分析
    print("\n" + "=" * 80)
    print("7️⃣ 完整性分析")
    print("=" * 80)

    print(f"\n原始代码总行数: {len(ref_lines)}")
    print(f"OCR 结果总行数: {len(ocr_lines)}")
    
    if len(ref_lines) > len(ocr_lines):
        print(f"\n⚠️ OCR 结果缺少 {len(ref_lines) - len(ocr_lines)} 行")
        print("\n缺失的内容 (最后几行):")
        for i in range(len(ocr_lines), len(ref_lines)):
            print(f"  第 {i+1} 行: {ref_lines[i]}")
    elif len(ocr_lines) > len(ref_lines):
        print(f"\n⚠️ OCR 结果多出 {len(ocr_lines) - len(ref_lines)} 行")
        print("\n多余的内容 (最后几行):")
        for i in range(len(ref_lines), len(ocr_lines)):
            print(f"  第 {i+1} 行: {ocr_lines[i]}")
    else:
        print("\n✅ 行数完全匹配")

    # 8. 总结
    print("\n" + "=" * 80)
    print("8️⃣ 评估总结")
    print("=" * 80)

    print(f"\n📊 指标汇总:")
    print(f"  - CER (字符错误率): {cer:.2f}%")
    print(f"  - WER (单词错误率): {wer:.2f}%")
    print(f"  - BLEU Score: {bleu * 100:.2f}")
    print(f"  - Exact Match Rate: {emr:.2f}%")
    
    print(f"\n📈 质量评价:")
    if cer < 5:
        print("  ✅ 优秀 - CER < 5%")
    elif cer < 10:
        print("  ✅ 良好 - CER < 10%")
    elif cer < 20:
        print("  ⚠️ 一般 - CER < 20%")
    else:
        print("  ❌ 较差 - CER >= 20%")
    
    if emr > 90:
        print("  ✅ 逐行匹配率优秀 (>90%)")
    elif emr > 80:
        print("  ✅ 逐行匹配率良好 (>80%)")
    else:
        print("  ⚠️ 逐行匹配率需改进 (<80%)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
