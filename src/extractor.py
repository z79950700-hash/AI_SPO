"""
LLM 三元组抽取模块
用 DeepSeek API 替代 CMeKG 的 Model4s + Model4po，实现零训练数据的关系抽取
"""

import json
import os
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

RELATIONS = ["is_a", "uses", "part_of", "applied_to", "improves", "compared_to", "proposed_by", "trained_on"]

EXTRACT_TOOL = {
    "type": "function",
    "function": {
        "name": "save_triples",
        "description": "保存从文本中抽取的AI概念知识三元组",
        "parameters": {
            "type": "object",
            "properties": {
                "triples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "s": {"type": "string", "description": "主语：AI技术概念，简洁（2-8字）"},
                            "p": {"type": "string", "enum": RELATIONS, "description": "关系类型"},
                            "o": {"type": "string", "description": "宾语：AI技术概念，简洁（2-8字）"}
                        },
                        "required": ["s", "p", "o"]
                    }
                }
            },
            "required": ["triples"]
        }
    }
}

SYSTEM_PROMPT = """你是AI概念知识图谱构建专家。从文本中抽取AI/ML技术概念之间的关系三元组。

关系说明：
- is_a：A是B的一种（BERT is_a 预训练语言模型）
- uses：A使用了B技术（Transformer uses 自注意力机制）
- part_of：A是B的组成部分（多头注意力 part_of Transformer）
- applied_to：A被用于B场景（BERT applied_to 文本分类）
- improves：A改进了B（GPT-2 improves GPT）
- compared_to：A与B存在对比关系（BERT compared_to GPT）
- proposed_by：A由B提出（Transformer proposed_by Vaswani）
- trained_on：A在B上训练（GPT trained_on 网络语料）

规则：
- 概念名称要标准化、简洁，不要长句子
- 只抽取有明确技术含义的三元组
- 同一概念统一用一种写法（统一用中文或英文，不要混用）"""


def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY，或在项目根目录创建 .env 文件")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def extract_triples(text: str, client: OpenAI) -> list[tuple[str, str, str]]:
    """
    从一段文本中抽取 (S, P, O) 三元组列表
    对标 CMeKG 的 extract_spoes(text, model4s, model4po)
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"从以下文本中抽取AI概念三元组：\n\n{text}"}
        ],
        tools=[EXTRACT_TOOL],
        tool_choice={"type": "function", "function": {"name": "save_triples"}},
    )
    tool_call = response.choices[0].message.tool_calls[0]
    result = json.loads(tool_call.function.arguments)
    return [(t["s"], t["p"], t["o"]) for t in result.get("triples", [])]


def extract_from_texts(texts: list[str], client: OpenAI) -> list[tuple[str, str, str]]:
    """
    批量处理多段文本，去重后返回所有三元组
    对标 CMeKG 的 get_triples(content, model4s, model4po)
    """
    seen = set()
    all_triples = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        print(f"[{i+1}/{len(texts)}] 抽取中：{text[:40].strip()}...")
        triples = extract_triples(text, client)
        for spo in triples:
            if spo not in seen:
                seen.add(spo)
                all_triples.append(spo)
        print(f"  → 本段 {len(triples)} 条，累计 {len(all_triples)} 条")
    return all_triples


def save_triples(triples: list[tuple[str, str, str]], path: str = "data/triples.json"):
    Path(path).parent.mkdir(exist_ok=True)
    data = [{"s": s, "p": p, "o": o} for s, p, o in triples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(triples)} 条三元组 → {path}")


def load_triples(path: str = "data/triples.json") -> list[tuple[str, str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(d["s"], d["p"], d["o"]) for d in data]
