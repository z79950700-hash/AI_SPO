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
    """初始化并返回 DeepSeek 客户端。

    Raises:
        ValueError: 如果未设置 DEEPSEEK_API_KEY 环境变量。

    Returns:
        OpenAI: 配置好的 OpenAI 客户端实例。
    """
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("请设置环境变量 DEEPSEEK_API_KEY，或在项目根目录创建 .env 文件")
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


def extract_triples(text: str, client: OpenAI) -> list[tuple[str, str, str]]:
    """从一段文本中抽取 (S, P, O) 三元组列表。

    对标 CMeKG 的 extract_spoes(text, model4s, model4po)。

    Args:
        text: 待抽取的文本。
        client: DeepSeek 客户端实例。

    Returns:
        list[tuple[str, str, str]]: 三元组列表，每个元素为 (主语, 关系, 宾语)。

    Raises:
        ValueError: 如果 API 响应中缺少 choices 或 tool_calls。
        json.JSONDecodeError: 如果返回的 JSON 格式错误。
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
    if not response.choices:
        raise ValueError("API 响应中没有返回 choices")
    message = response.choices[0].message
    if not message.tool_calls:
        raise ValueError("模型没有返回 tool_calls，可能是输出格式错误")
    tool_call = message.tool_calls[0]
    try:
        result = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"解析函数参数 JSON 失败: {e}", e.doc, e.pos) from e
    triples = result.get("triples", [])
    if not triples:
        return []
    return [(t["s"], t["p"], t["o"]) for t in triples]


def extract_from_texts(texts: list[str], client: OpenAI) -> list[tuple[str, str, str]]:
    """批量处理多段文本，去重后返回所有三元组。

    对标 CMeKG 的 get_triples(content, model4s, model4po)。

    Args:
        texts: 多段文本的列表。
        client: DeepSeek 客户端实例。

    Returns:
        list[tuple[str, str, str]]: 去重后的所有三元组列表。
    """
    seen: set[tuple[str, str, str]] = set()
    all_triples: list[tuple[str, str, str]] = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        print(f"[{i+1}/{len(texts)}] 抽取中：{text[:40].strip()}...")
        try:
            triples = extract_triples(text, client)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"  \u2717 抽取失败: {e}")
            continue
        for spo in triples:
            if spo not in seen:
                seen.add(spo)
                all_triples.append(spo)
        print(f"  \u2192 本段 {len(triples)} 条，累计 {len(all_triples)} 条")
    return all_triples


def save_triples(triples: list[tuple[str, str, str]], path: str = "data/triples.json") -> None:
    """将三元组列表保存到 JSON 文件。

    Args:
        triples: 三元组列表，每个元素为 (主语, 关系, 宾语)。
        path: 输出文件路径，默认为 "data/triples.json"。
    """
    Path(path).parent.mkdir(exist_ok=True)
    data = [{"s": s, "p": p, "o": o} for s, p, o in triples]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"已保存 {len(triples)} 条三元组 \u2192 {path}")


def load_triples(path: str = "data/triples.json") -> list[tuple[str, str, str]]:
    """从 JSON 文件加载三元组列表。

    Args:
        path: 输入文件路径，默认为 "data/triples.json"。

    Returns:
        list[tuple[str, str, str]]: 三元组列表。

    Raises:
        FileNotFoundError: 如果文件不存在。
        json.JSONDecodeError: 如果文件内容不是有效的 JSON。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件不存在: {path}")
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"JSON 解析失败: {e}", e.doc, e.pos) from e
    return [(d["s"], d["p"], d["o"]) for d in data]
