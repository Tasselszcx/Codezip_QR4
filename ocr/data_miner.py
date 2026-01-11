import json
import time
import random
from datetime import datetime
from github import Github
from tqdm import tqdm

# ================= 配置区 (根据您的要求修改) =================
# ⚠️ 请务必在此处填入您的 GitHub Token
GITHUB_TOKEN = "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" 

TARGET_DATE = "2025-08-01"  # 截止日期
TARGET_LANG = "python"      # 目标语言
MIN_STARS = 50              # 最小 Star 数
MAX_STARS = 200             # 最大 Star 数
MIN_LINES = 50              # 最小行数
MAX_LINES = 120             # 最大行数
LIMIT = 10                  # 抓取数量
OUTPUT_FILE = "dataset_fresh_2025.json"

# 随机化设置
ENABLE_RANDOM = True        # 是否启用随机化
RANDOM_POOL_SIZE = 50       # 从前 N 个结果中随机抽取
# =========================================================

def fetch_fresh_code():
    # 简单的 Token 检查
    if "ghp_" not in GITHUB_TOKEN and "github_" not in GITHUB_TOKEN:
        print("⚠️ 警告: GitHub Token 可能未配置，请检查 data_miner.py")

    print(f"🚀 [Module 1] Data Miner Started...")
    print(f"📅 Filter: Created > {TARGET_DATE} | Lines: {MIN_LINES}-{MAX_LINES} | Limit: {LIMIT}")
    
    g = Github(GITHUB_TOKEN)
    
    # 随机化查询参数
    if ENABLE_RANDOM:
        # 随机选择排序方式和顺序
        sort_options = ["stars", "forks", "updated"]
        order_options = ["desc", "asc"]
        sort_by = random.choice(sort_options)
        order_by = random.choice(order_options)
        
        # 随机偏移星星范围 (在 MIN_STARS~MAX_STARS 基础上随机偏移)
        star_offset = random.randint(0, 50)
        actual_min_stars = MIN_STARS + star_offset
        actual_max_stars = MAX_STARS + star_offset
        
        print(f"🎲 Random mode: sort={sort_by}, order={order_by}, stars={actual_min_stars}..{actual_max_stars}")
    else:
        sort_by = "stars"
        order_by = "desc"
        actual_min_stars = MIN_STARS
        actual_max_stars = MAX_STARS
    
    query = f"language:{TARGET_LANG} created:>{TARGET_DATE} stars:{actual_min_stars}..{actual_max_stars}"
    
    try:
        repos = g.search_repositories(query, sort=sort_by, order=order_by)
    except Exception as e:
        print(f"❌ GitHub API Error: {e}")
        return []

    # 收集候选仓库（先收集一个池子，再随机抽取）
    candidate_repos = []
    repo_count = 0
    
    print(f"📦 Building candidate pool (max {RANDOM_POOL_SIZE} repos)...")
    for repo in repos:
        if repo_count >= RANDOM_POOL_SIZE:
            break
        candidate_repos.append(repo)
        repo_count += 1
        time.sleep(0.05)  # 避免 API 限制
    
    # 随机打乱候选仓库顺序
    if ENABLE_RANDOM:
        random.shuffle(candidate_repos)
        print(f"🔀 Shuffled {len(candidate_repos)} candidate repos")

    dataset = []
    pbar = tqdm(total=LIMIT, desc="Mining Code")

    for repo in candidate_repos:
        if len(dataset) >= LIMIT:
            break
        try:
            contents = repo.get_contents("")
            files_to_check = []
            while contents:
                file_content = contents.pop(0)
                if file_content.type == "dir":
                    if file_content.path in ['src', 'lib', 'core', 'app']:
                        try:
                            contents.extend(repo.get_contents(file_content.path))
                        except: pass
                elif file_content.path.endswith(".py"):
                    if "test" not in file_content.path and "__init__" not in file_content.path:
                        files_to_check.append(file_content)
            
            for file_node in files_to_check:
                if 1000 < file_node.size < 20000:
                    code_text = file_node.decoded_content.decode('utf-8')
                    lines = code_text.splitlines()
                    if MIN_LINES <= len(lines) <= MAX_LINES:
                        dataset.append({
                            "id": f"{repo.name}_{file_node.path}".replace("/", "_"), # 扁平化ID方便做文件名
                            "repo": repo.full_name,
                            "url": file_node.html_url,
                            "code": code_text,
                            "line_count": len(lines)
                        })
                        pbar.update(1)
                        break 
        except:
            continue
        time.sleep(0.1)

    pbar.close()
    
    print(f"✅ [Module 1] Completed. Saved {len(dataset)} items to {OUTPUT_FILE}")
    return dataset

if __name__ == "__main__":
    fetch_fresh_code()