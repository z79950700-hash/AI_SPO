"""
图构建与可视化模块
输入：(S, P, O) 三元组列表
输出：knowledge_graph.html（可在浏览器直接打开）
"""

from pathlib import Path
from pyvis.network import Network
import networkx as nx

from .extractor import load_triples

RELATION_COLORS = {
    "is_a":        "#E74C3C",
    "uses":        "#3498DB",
    "part_of":     "#2ECC71",
    "applied_to":  "#F39C12",
    "improves":    "#9B59B6",
    "compared_to": "#1ABC9C",
    "proposed_by": "#95A5A6",
    "trained_on":  "#E67E22",
}


def build_graph(triples: list[tuple[str, str, str]]) -> nx.DiGraph:
    """从三元组列表构建有向图。

    Args:
        triples: 三元组列表，每个元素为 (主语, 关系, 宾语)。

    Returns:
        nx.DiGraph: 构建好的有向图对象。
    """
    G = nx.DiGraph()
    for s, p, o in triples:
        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, relation=p)
    return G


def visualize(G: nx.DiGraph, output: str = "knowledge_graph.html") -> None:
    """将图可视化为交互式 HTML 文件。

    Args:
        G: 要可视化的有向图。
        output: 输出 HTML 文件路径。
    """
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#1a1a2e",
        font_color="white",
        notebook=False,
    )

    degree = dict(G.degree())
    for node in G.nodes():
        size = 15 + degree[node] * 5
        net.add_node(
            node,
            label=node,
            size=min(size, 60),
            color="#4ECDC4",
            font={"size": 14, "color": "white"},
            title=f"{node}\n连接数：{degree[node]}",
        )

    for s, o, data in G.edges(data=True):
        relation = data.get("relation", "")
        color = RELATION_COLORS.get(relation, "#888888")
        net.add_edge(
            s, o,
            label=relation,
            color=color,
            title=relation,
            arrows="to",
            font={"size": 11, "color": color},
        )

    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "springLength": 150,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    net.write_html(output)
    print(f"图谱已生成：{output}，用浏览器打开即可查看")


def print_stats(G: nx.DiGraph) -> None:
    """打印图谱统计信息。

    Args:
        G: 要统计的有向图。
    """
    print(f"\n图谱统计：")
    print(f"  节点数（概念）：{G.number_of_nodes()}")
    print(f"  边数（关系）：  {G.number_of_edges()}")

    relation_counts = {}
    for _, _, d in G.edges(data=True):
        r = d.get("relation", "unknown")
        relation_counts[r] = relation_counts.get(r, 0) + 1
    print("  关系分布：")
    for r, cnt in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"    {r}: {cnt}")

    top_nodes = sorted(G.degree(), key=lambda x: -x[1])[:10]
    print("  最核心概念 Top10：")
    for node, deg in top_nodes:
        print(f"    {node}: {deg} 条关系")


if __name__ == "__main__":
    # 简单测试：从默认路径加载并生成图谱
    try:
        triples = load_triples()
        if triples:
            G = build_graph(triples)
            print_stats(G)
            visualize(G)
        else:
            print("没有加载到三元组。")
    except FileNotFoundError as e:
        print(f"{e}\n请先运行 extractor 模块生成三元组数据。")
    except Exception as e:
        print(f"发生错误: {e}")
