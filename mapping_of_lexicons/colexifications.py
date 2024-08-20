import os
import sys
import json
import networkx as nx
import matplotlib.pyplot as plt

from mapping_of_lexicons.graph_utils import get_inverse_dic

def get_colexification_graph(dic):
    inverse_dic = get_inverse_dic(dic)
    G = nx.DiGraph()
    source_keys = list(dic.keys())
    G.add_nodes_from(source_keys)
    for source_key in source_keys:
        colexifications = get_colexifications(dic, inverse_dic, source_key)
        G.add_edges_from({(source_key, colexification) for colexification in colexifications if source_key != colexification})
    return G


def get_colexifications(dic, inverse_dic, word):
    colexifications = []
    if word in dic.keys():
        for target_value in dic[word].keys():
            colexifications.extend(list(inverse_dic[target_value].keys()))
    return colexifications


def plot_graph(G):
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700)
    plt.show()


if __name__ == '__main__':
    with open(r"D:\python project\data\alignments\en-es_alignments\en-es_alignments_lemmas_only1to1_clean_inverse_clean_inverse.json", "r") as json_file:
            lemmas_dic = json.load(json_file)
    words_to_display = ["about", "above", "across", "after", "against", "along", "among", "around", "at", "before",
                       "behind", "below", "beneath", "beside", "between", "beyond", "but", "by", "concerning",
                       "considering", "despite", "down", "during", "except", "for", "from", "in", "inside", "into",
                       "like", "near", "of", "off", "on", "onto", "out", "outside", "over", "past", "regarding",
                       "round", "since", "through", "throughout", "to", "toward", "under", "underneath", "until",
                       "unto", "up", "upon", "with", "within", "without"]
    lemmas_dic = {key: lemmas_dic[key] for key in words_to_display if key in lemmas_dic}

    G = get_colexification_graph(lemmas_dic)
    plot_graph(G)
    print("done")
