"""
自动优化循环：push PR → Greptile GitHub App 自动评审 → 读 PR 评论拿评分
          → DeepSeek 修复 → 推更新 → 循环直到评分 ≥ 4

前提：Greptile GitHub App 已安装在仓库上
从项目根目录运行：python scripts/improve.py
"""

import subprocess
import requests
import time
import json
import re
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / ".env")

REPO = "z79950700-hash/AI_SPO"
BASE_BRANCH = "main"
REVIEW_BRANCH = "greptile-review"
SOURCE_FILES = ["src/extractor.py", "src/graph.py"]


# ─── GitHub API ─────────────────────────────────────────────────────────────

def _gh_headers() -> dict:
    return {
        "Authorization": f"token {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


# ─── Git ────────────────────────────────────────────────────────────────────

def setup_review_branch():
    """切换到 greptile-review 分支（本地不存在则创建）"""
    local = subprocess.run(
        ["git", "branch", "--list", REVIEW_BRANCH],
        capture_output=True, text=True,
    )
    if local.stdout.strip():
        subprocess.run(["git", "checkout", REVIEW_BRANCH], check=True)
    else:
        subprocess.run(["git", "checkout", "-b", REVIEW_BRANCH], check=True)
    print(f"当前分支：{REVIEW_BRANCH}")


def git_push(commit_msg: str, force: bool = False) -> bool:
    """提交并推送。force=True 时即使无新 commit 也 push（用于首次建立远端分支）。
    返回是否产生了新 commit（只有新 commit 才会触发 Greptile 评审）。"""
    subprocess.run(["git", "add"] + SOURCE_FILES + ["scripts/improve.py"], check=True)
    result = subprocess.run(["git", "commit", "-m", commit_msg])
    committed = result.returncode == 0
    if not committed and not force:
        print("没有新变更，跳过 push")
        return False
    subprocess.run(["git", "push", "-u", "origin", REVIEW_BRANCH], check=True)
    print(f"已推送：{commit_msg}")
    return committed


# ─── PR 管理 ─────────────────────────────────────────────────────────────────

def ensure_pr() -> int:
    """确保存在 greptile-review → main 的 PR，返回 PR 号"""
    owner = REPO.split("/")[0]
    resp = requests.get(
        f"https://api.github.com/repos/{REPO}/pulls",
        headers=_gh_headers(),
        params={"head": f"{owner}:{REVIEW_BRANCH}", "base": BASE_BRANCH, "state": "open"},
    )
    resp.raise_for_status()
    pulls = resp.json()
    if pulls:
        pr_number = pulls[0]["number"]
        print(f"复用已有 PR #{pr_number}")
        return pr_number

    resp = requests.post(
        f"https://api.github.com/repos/{REPO}/pulls",
        headers=_gh_headers(),
        json={
            "title": "AI_SPO: Greptile + DeepSeek 自动优化",
            "body": "由 improve.py 自动创建，循环优化直到 Greptile 评分 ≥ 4/5",
            "head": REVIEW_BRANCH,
            "base": BASE_BRANCH,
        },
    )
    resp.raise_for_status()
    pr_number = resp.json()["number"]
    print(f"已创建 PR #{pr_number}")
    return pr_number


# ─── 读取 Greptile 评审 ───────────────────────────────────────────────────────

def _parse_score(body: str) -> float | None:
    """从 Greptile 评审正文提取数字评分（1-5）"""
    patterns = [
        r'(?:confidence|score)[:\s]+(\d+(?:\.\d+)?)\s*/\s*5',
        r'(\d+(?:\.\d+)?)\s*/\s*5\s+confidence',
        r'\*\*(\d+(?:\.\d+)?)/5\*\*',
        r'(\d+(?:\.\d+)?)\s*/\s*5',
    ]
    for pat in patterns:
        m = re.search(pat, body, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def wait_for_greptile_review(pr_number: int, since_time: str) -> tuple[float, str]:
    """轮询 PR，等待 Greptile 在 since_time 之后发布评审。
    返回 (score, raw_body)，raw_body 是评审原文供 DeepSeek 参考。"""
    print("等待 Greptile 评审（最多 10 分钟）...")
    last_greptile_body = None

    for i in range(40):
        time.sleep(15)

        # 查 PR reviews
        rev_resp = requests.get(
            f"https://api.github.com/repos/{REPO}/pulls/{pr_number}/reviews",
            headers=_gh_headers(),
        )
        for review in reversed(rev_resp.json() or []):
            login = review.get("user", {}).get("login", "")
            submitted_at = review.get("submitted_at", "")
            body = review.get("body", "")
            if "greptile" in login.lower() and submitted_at > since_time and body:
                last_greptile_body = body
                score = _parse_score(body)
                if score is not None:
                    print("  找到评分（来自 PR review）")
                    return score, body

        # 查 issue comments
        cmt_resp = requests.get(
            f"https://api.github.com/repos/{REPO}/issues/{pr_number}/comments",
            headers=_gh_headers(),
        )
        for comment in reversed(cmt_resp.json() or []):
            login = comment.get("user", {}).get("login", "")
            # Greptile 编辑同一条评论而非新建，用 updated_at 而非 created_at
            updated_at = comment.get("updated_at", "")
            body = comment.get("body", "")
            if "greptile" in login.lower() and updated_at > since_time and body:
                last_greptile_body = body
                score = _parse_score(body)
                if score is not None:
                    print("  找到评分（来自 PR comment）")
                    return score, body

        if last_greptile_body:
            print(f"  [{i+1}] Greptile 已评论但未解析到评分，原文：{last_greptile_body[:200]}")
        else:
            print(f"  [{i+1}] 等待 Greptile 评审...")

    msg = "等待超时（10 分钟）"
    if last_greptile_body:
        msg += f"\n\n【Greptile 原文，请告诉我如何提取评分】\n{last_greptile_body[:800]}"
    else:
        msg += f"\n请检查：\n1. Greptile GitHub App 是否正确安装\n2. PR #{pr_number} 是否存在"
    raise TimeoutError(msg)


# ─── 代码修复 ────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    """按花括号深度找第一个完整的 JSON 对象，避免贪婪正则截断嵌套结构。"""
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == '{':
            if start is None:
                start = i
            depth += 1
        elif ch == '}' and start is not None:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    start = None  # 这段不合法，继续往后找
    return None


def fix_code(raw_review: str, client: OpenAI):
    files_content = {
        f: Path(f).read_text(encoding="utf-8")
        for f in SOURCE_FILES
        if Path(f).exists()
    }
    files_text = "\n\n".join(f"=== {name} ===\n{code}" for name, code in files_content.items())

    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=4096,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是Python代码优化专家。根据代码评审内容改进代码，保持原有功能不变。"
                    '只返回JSON，格式：{"文件名": "完整修改后的代码"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"以下是 Greptile 对这份 PR 的评审原文：\n{raw_review}\n\n"
                    f"当前代码：\n{files_text}\n\n"
                    "请根据评审内容，找出可以实际改进的地方（错误处理、文档、代码规范、类型注解等），"
                    "修改代码。只返回JSON：{\"文件名\": \"完整修改后的代码\"}"
                ),
            },
        ],
    )

    raw = response.choices[0].message.content
    result = _extract_json(raw)
    if result is None:
        print(f"DeepSeek 返回格式异常，跳过本轮修改\n原文：{raw[:300]}")
        return
    for fname, new_content in result.items():
        if fname in SOURCE_FILES:
            Path(fname).write_text(new_content, encoding="utf-8")
            print(f"已更新：{fname}")


# ─── 主循环 ──────────────────────────────────────────────────────────────────

def run_loop(max_iter: int = 5):
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    setup_review_branch()

    # 首次推送，触发第一次 Greptile 评审（force=True 确保远端分支存在）
    since_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    git_push("improve: start greptile review loop", force=True)
    pr_number = ensure_pr()

    for i in range(max_iter):
        print(f"\n{'='*50}")
        print(f"第 {i+1} 轮评审")
        print('='*50)

        score, raw_body = wait_for_greptile_review(pr_number, since_time)

        print(f"当前评分：{score:.1f} / 5.0")

        if score >= 4:
            print(f"\n✅ 评分达标（{score}/5），优化完成！")
            break

        print(f"\n评分 {score} < 4，开始修改...")
        fix_code(raw_body, client)

        since_time = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        pushed = git_push(f"improve: round {i+1}, prev score={score:.1f}")
        if not pushed:
            print("代码无实质变化，DeepSeek 未能从评审中提取改进点，退出")
            break
    else:
        print(f"\n⚠️ 已达最大循环次数（{max_iter}轮），停止")


if __name__ == "__main__":
    run_loop()
