"""
自动优化循环：push 代码 → Greptile 评审 → DeepSeek 修复 → 循环直到评分 > 4
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

# 始终从项目根目录运行，无论从哪里启动脚本
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)
load_dotenv(PROJECT_ROOT / ".env")

REPO = "z79950700-hash/AI_SPO"
BRANCH = "main"
SOURCE_FILES = ["src/extractor.py", "src/graph.py"]


# ─── Git ────────────────────────────────────────────────────────────────────

def git_init_and_push_first():
    if not Path(".git").exists():
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "branch", "-M", BRANCH], check=True)
        token = os.environ["GITHUB_TOKEN"]
        remote = f"https://{token}@github.com/{REPO}.git"
        subprocess.run(["git", "remote", "add", "origin", remote], check=True)
        print("Git 仓库已初始化")
    git_push("feat: initial commit")


def git_push(commit_msg: str):
    subprocess.run(["git", "add", "."], check=True)
    result = subprocess.run(["git", "commit", "-m", commit_msg])
    if result.returncode != 0:
        print("没有新变更，跳过 commit")
        return
    subprocess.run(["git", "push", "-u", "origin", BRANCH], check=True)
    print(f"已推送：{commit_msg}")


# ─── Greptile ───────────────────────────────────────────────────────────────

def _greptile_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['GREPTILE_API_KEY']}",
        "X-Github-Token": os.environ["GITHUB_TOKEN"],
        "Content-Type": "application/json",
    }


def _greptile_mcp_headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['GREPTILE_API_KEY']}",
        "Content-Type": "application/json",
    }


def index_repo():
    resp = requests.post(
        "https://api.greptile.com/v2/repositories",
        headers=_greptile_headers(),
        json={"remote": "github", "repository": REPO, "branch": BRANCH},
    )
    resp.raise_for_status()
    print("Greptile 索引已触发，轮询状态...")

    # 轮询直到索引完成，repositoryId 格式：github:main:owner%2Frepo
    repo_id = f"github:{BRANCH}:{REPO.replace('/', '%2F')}"
    for i in range(20):  # 最多等 5 分钟
        time.sleep(15)
        status_resp = requests.get(
            f"https://api.greptile.com/v2/repositories/{repo_id}",
            headers=_greptile_headers(),
        )
        if status_resp.status_code == 200:
            status = status_resp.json().get("status", "")
            print(f"  [{i+1}] 索引状态：{status}")
            if status.upper() == "COMPLETED":
                print("索引完成！")
                return
        else:
            print(f"  [{i+1}] 等待中...")

    print("索引超时，尝试继续...")


def get_review() -> tuple[float, list[str]]:
    prompt = (
        "请对这个仓库的代码质量打分（1-5分），"
        "从代码规范、错误处理、可维护性、文档注释四个维度综合评估。"
        '严格返回 JSON，不要包含其他文字：{"score": 数字, "suggestions": ["建议1", "建议2"]}'
    )

    # 第一步：探测 MCP 支持的方法列表
    list_resp = requests.post(
        "https://api.greptile.com/mcp",
        headers=_greptile_mcp_headers(),
        json={"jsonrpc": "2.0", "id": 0, "method": "tools/list", "params": {}},
    )
    print(f"  tools/list 响应 {list_resp.status_code}：{list_resp.text[:400]}")

    # 第二步：根据探测结果决定用哪个方法名
    tool_method = "tools/call"  # 默认 MCP 标准方法
    if list_resp.ok:
        list_result = list_resp.json()
        if "error" in list_result:
            # MCP 不支持 tools/list，尝试直接用 query
            tool_method = "query"

    mcp_payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": tool_method,
        "params": {
            "name": "query",
            "arguments": {
                "query": prompt,
                "repositories": [{"remote": "github", "repository": REPO, "branch": BRANCH}],
            },
        },
    }
    resp = requests.post(
        "https://api.greptile.com/mcp",
        headers=_greptile_mcp_headers(),
        json=mcp_payload,
    )
    print(f"  MCP {tool_method} 响应 {resp.status_code}：{resp.text[:400]}")

    mcp_result = resp.json() if resp.ok else {}
    if resp.ok and "error" not in mcp_result:
        content = mcp_result.get("result", {}).get("content", [])
        raw = content[0].get("text", "") if content else str(mcp_result.get("result", ""))
    else:
        # 降级：旧 REST API
        print("  MCP 不可用，降级到 REST /v2/query ...")
        rest_payload = {
            "messages": [{"id": "1", "role": "user", "content": prompt}],
            "repositories": [{"remote": "github", "repository": REPO, "branch": BRANCH}],
            "stream": False,
            "genius": True,
        }
        resp2 = requests.post(
            "https://api.greptile.com/v2/query",
            headers=_greptile_headers(),
            json=rest_payload,
        )
        if not resp2.ok:
            raise RuntimeError(
                f"MCP 失败：{resp.text[:200]}\n"
                f"REST 失败 {resp2.status_code}：{resp2.text[:200]}"
            )
        raw = resp2.json().get("message", "") or str(resp2.json())

    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        raise ValueError(f"Greptile 返回格式异常：{raw}")
    data = json.loads(match.group())
    return float(data["score"]), data["suggestions"]


# ─── 代码修复 ────────────────────────────────────────────────────────────────

def fix_code(suggestions: list[str], client: OpenAI):
    files_content = {
        f: Path(f).read_text(encoding="utf-8")
        for f in SOURCE_FILES
        if Path(f).exists()
    }
    files_text = "\n\n".join(f"=== {name} ===\n{code}" for name, code in files_content.items())
    suggestions_text = "\n".join(f"- {s}" for s in suggestions)

    response = client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=4096,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是Python代码优化专家。根据改进建议修改代码，保持原有功能不变。"
                    '只返回JSON，格式：{"文件名": "完整修改后的代码"}'
                ),
            },
            {
                "role": "user",
                "content": f"改进建议：\n{suggestions_text}\n\n当前代码：\n{files_text}",
            },
        ],
    )

    raw = response.choices[0].message.content
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not match:
        print(f"DeepSeek 返回格式异常，跳过本轮修改")
        return

    result = json.loads(match.group())
    for fname, new_content in result.items():
        if fname in SOURCE_FILES:
            Path(fname).write_text(new_content, encoding="utf-8")
            print(f"已更新：{fname}")


# ─── 主循环 ──────────────────────────────────────────────────────────────────

def run_loop(max_iter: int = 10):
    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

    git_init_and_push_first()
    index_repo()

    for i in range(max_iter):
        print(f"\n{'='*50}")
        print(f"第 {i+1} 轮评审")
        print('='*50)

        score, suggestions = get_review()
        print(f"当前评分：{score:.1f} / 5.0")
        print("Greptile 建议：")
        for s in suggestions:
            print(f"  · {s}")

        if score >= 4:
            print(f"\n✅ 评分达标（{score}/5），优化完成！")
            break

        print(f"\n评分 {score} < 4，开始修改...")
        fix_code(suggestions, client)
        git_push(f"improve: round {i+1}, prev score={score:.1f}")

        if i < max_iter - 1:
            print("等待 Greptile 重新索引（60秒）...")
            time.sleep(60)
    else:
        print(f"\n⚠️ 已达最大循环次数（{max_iter}轮），停止")


if __name__ == "__main__":
    run_loop()
