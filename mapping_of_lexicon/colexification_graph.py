import os
import sys
import json
import networkx as nx
import matplotlib.pyplot as plt

sys.path.insert(0, r"C:\Users\RoyIlani\pythonProject\Lexical_semantics_in_cross-lingual_transfer")
from transformations.semantic_transformation import get_inverse_dic


def get_colexification_graph(dic):
    inverse_dic = get_inverse_dic(dic)
    G = nx.DiGraph()
    source_keys = list(dic.keys())
    G.add_nodes_from(source_keys)
    for source_key in source_keys:
        for target_value in dic[source_key].keys():
            colexifications = inverse_dic[target_value].keys()
            G.add_edges_from({(source_key, colexification) for colexification in colexifications})
    return G


def plot_graph(G):
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700)
    plt.show()


if __name__ == '__main__':
    main_dir = ""
    with open(os.path.join(main_dir, "en-es.json"), "r") as json_file:
            lemmas_dic = json.load(json_file)
    lemmas_dic = dict(list(lemmas_dic.items())[:20])

    G = get_colexification_graph(lemmas_dic)
    plot_graph(G)
    print("done")
