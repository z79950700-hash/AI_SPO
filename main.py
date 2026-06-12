"""
项目入口：抽取三元组 → 构建知识图谱 → 可视化
"""

from src.extractor import get_client, extract_from_texts, save_triples, load_triples
from src.graph import build_graph, print_stats, visualize

TEXTS = [
    "BERT是由Google提出的预训练语言模型，基于Transformer的Encoder部分，使用双向注意力机制，在大规模语料上进行MLM和NSP两个预训练任务。",
    "GPT系列模型使用Transformer的Decoder部分，采用自回归方式生成文本，GPT-2在GPT基础上扩大了参数量，GPT-3进一步扩展到1750亿参数。",
    "Transformer由Vaswani等人在2017年提出，核心是Multi-Head Self-Attention机制，由Encoder和Decoder两部分组成，完全取代了RNN结构。",
]


def main():
    client = get_client()
    triples = extract_from_texts(TEXTS, client)
    save_triples(triples)

    G = build_graph(triples)
    print_stats(G)
    visualize(G)


if __name__ == "__main__":
    main()
