import os
import sys
import json
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\RoyIlani\pythonProject\Lexical_semantics_in_cross-lingual_transfer")
from transformations.semantic_transformation import get_inverse_dic


def get_colexification_graph(dic):
    inverse_dic = get_inverse_dic(dic)
    graph_vertices = list(dic.keys())
    graph_edges = set()
    for vertix in graph_vertices:
        for target_value in dic[vertix].keys():
            colexifications = inverse_dic[target_value].keys()
            edges = {(graph_vertices.index(vertix), graph_vertices.index(colexification))
                     for colexification in colexifications}
            graph_edges.update(edges)
    return graph_vertices, graph_edges


def plot_graph(graph_vertices, graph_edges):
    G = nx.DiGraph()
    G.add_nodes_from(graph_vertices)
    G.add_edges_from(graph_edges)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700)
    plt.show()


if __name__ == '__main__':
    main_dir = ""
    with open(os.path.join(main_dir, "en-es.json"), "r") as json_file:
            lemmas_dic = json.load(json_file)

    graph_vertices, graph_edges = get_colexification_graph(lemmas_dic)
    plot_graph(graph_vertices, graph_edges)
    print("done")
