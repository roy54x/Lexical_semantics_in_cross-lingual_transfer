import os
import sys
import json
import networkx as nx
import matplotlib.pyplot as plt

from mapping_of_lexicons.graph_utils import get_inverse_dic

sys.path.insert(0, r"C:\Users\RoyIlani\pythonProject\Lexical_semantics_in_cross-lingual_transfer")



def get_colexification_graph(dic):
    inverse_dic = get_inverse_dic(dic)
    G = nx.DiGraph()
    source_keys = list(dic.keys())
    G.add_nodes_from(source_keys)
    for source_key in source_keys:
        colexifications = get_colexifications(dic, inverse_dic, source_key)
        G.add_edges_from({(source_key, colexification) for colexification in colexifications})
    return G


def get_colexifications(dic, inverse_dic, word):
    colexifications = []
    for target_value in dic[word].keys():
        colexifications.extend(list(inverse_dic[target_value].keys()))
    return colexifications


def plot_graph(G):
    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_size=700)
    plt.show()


if __name__ == '__main__':
    main_dir = ""
    with open(os.path.join(main_dir, "en-zh_cn.json"), "r") as json_file:
            lemmas_dic = json.load(json_file)
    inverse_dic = get_inverse_dic(lemmas_dic)
    words_to_display = ["about", "above", "across", "after", "against", "along", "among", "around", "at", "before",
                       "behind", "below", "beneath", "beside", "between", "beyond", "but", "by", "concerning",
                       "considering", "despite", "down", "during", "except", "for", "from", "in", "inside", "into",
                       "like", "near", "of", "off", "on", "onto", "out", "outside", "over", "past", "regarding",
                       "round", "since", "through", "throughout", "to", "toward", "under", "underneath", "until",
                       "unto", "up", "upon", "with", "within", "without"]
    keys_to_extract = set()
    for word in words_to_display:
        keys_to_extract.update(get_colexifications(lemmas_dic, inverse_dic, word))
    lemmas_dic = {key: lemmas_dic[key] for key in keys_to_extract if key in lemmas_dic}

    G = get_colexification_graph(lemmas_dic)
    plot_graph(G)
    print("done")
