# Lexical Semantics in Cross-Lingual Transfer

This repository provides the code to map lexicons of two languages, as discussed in [Assessing the Role of Lexical Semantics in Cross-lingual Transfer through Controlled Manipulations](https://arxiv.org/abs/2408.07599v1). The main functionalities include extracting bipartite graphs, calculating translation entropy, and generating colexification graphs.

## Table of Contents
1. [Installation](#installation)
2. [Download Bitext](#Download-bitext)
3. [Extracting the Bipartite Graph](#extracting-the-bipartite-graph)
4. [Filtering the Graph](#filtering-the-graph)
5. [Calculating Translation Entropy](#calculating-translation-entropy)
6. [Generating a Colexification Graph](#generating-a-colexification-graph)
7. [References](#references)

## Installation

First, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/roy54x/Lexical_Semantics_in_Cross-Lingual_Transfer.git
cd Lexical_Semantics_in_Cross-Lingual_Transfer
pip install -r requirements.txt
```

## Download Bitext

Download a parallel corpus for the two languages you wish to map their lexicons. For example, if you want to map two European languages, you can use the [Europarl](https://www.statmt.org/europarl/) corpus.

## Extracting the Bipartite Graph
<img src="https://github.com/user-attachments/assets/6e755d32-4ab2-44d5-a05a-f014446f2cb3" alt="Bipartite Graph Example" width="400">

### Explanation
The bipartite graph represents the mapping between the lexicons of two languages, denoted as $L_s$ (source language) and $L_t$ (target language). Formally, we define a weighted bipartite graph $G = (V_s, V_t, E, w)$, where:

- $V_s$ is the set of words in the lexicon of $L_s$.
- $V_t$ is the set of words in the lexicon of $L_t$.
- $E$ is the set of edges where a pair $(v, u) \in V_s \times V_t$ exists if the word $v \in L_s$ is aligned with the word $u \in L_t$ in at least one instance in the bitext.
- $w:E \rightarrow \mathbb{N}^+$ is a weight function assigning the number of times each word pair is aligned in the bitext.

### Code Example
To extract the bipartite graph that maps the two lexicons, use the following code:

```python
import os
from utils.utils import load_doc, parse_sentences_from_doc
from mapping_of_lexicons.create_bipartite_graph import create_bipartite_graph

main_dir = r"data/es-en" # Replace with the main directory of the bitext
source_doc_name, target_doc_name = "europarl-v7.es-en.en", "europarl-v7.es-en.es"

source_doc = load_doc(os.path.join(main_dir, source_doc_name))
target_doc = load_doc(os.path.join(main_dir, target_doc_name))
source_sentences = parse_sentences_from_doc(source_doc)
target_sentences = parse_sentences_from_doc(target_doc)

create_bipartite_graph(source_sentences, target_sentences, source_language="english", target_language="spanish",
                           output_file_name="europarl_en-es_alignments")
```

## Filtering the Graph

After we extract the graph, we filter out edges whose weights do not exceed a certain threshold or are relatively small compared to other edges originating from the same vertex. The filtering can be done with the following code:

```python
from mapping_of_lexicons.graph_utils import get_clean_dic, get_inverse_dic

file_name = "europarl_en-es_alignments" # Replace with the file name
file_path = os.path.join(main_dir, file_name + ".json")
with open(file_path, "r") as json_file:
    graph = json.load(json_file)

min_amount = 5
min_precent = 0.02
clean_graph = get_clean_dic(graph, min_amount, min_precent) # filter edges based on source vertices
clean_graph = get_inverse_dic(get_clean_dic(get_inverse_dic(clean_graph), min_amount, min_precent)) # filter edges based on target vertices

with open(file_path, "w") as outfile:
    json.dump(clean_graph, outfile)
```

## Calculating Translation Entropy
<img src="https://github.com/user-attachments/assets/715476e6-a699-4c80-b112-b18c208161eb" alt="Bipartite Graph Example" width="400">

### Explanation
To further appreciate the impact of the divergence between the source and the target lexicons, we introduce the concept of *translation entropy*. Let $G$ be the weighted bipartite graph presented earlier, we compute the entropy for each vertex $v$ in the graph:

<img src="https://github.com/user-attachments/assets/85471caf-3056-4b6a-bdf7-e7badb440f2a" alt="Bipartite Graph Example" width="250">

where $U_{v}$ is the subset of vertices linked to $v$, and $p_{v}$ is the following probability function:

<img src="https://github.com/user-attachments/assets/495e9700-4634-4ef3-87a8-dbd9acdaf01d" alt="Bipartite Graph Example" width="250">

### Code Example

To extract the translation entopy values, use: 

```python
from mapping_of_lexicons.graph_utils import get_entropies

get_entropies(main_dir, file_name, to_json=False)
```
This code will produce a csv file such as: 

<img src="https://github.com/user-attachments/assets/e00b82f4-2fc7-401f-959f-538fa984d2dd" alt="Bipartite Graph Example" width="300">

## Generating a Colexification Graph

Additionally, the bipartite graph can be used to generate a colexification graph. Colexification refers to the phenomenon where different meanings are expressed by the same word in a language ([wikipedia](https://en.wikipedia.org/wiki/Colexification)), and a colexification graph captures these relationships by connecting words that share meanings across different contexts. The mapping of the two lexicons enable us to identify words in the source language which share the same translation in the target language, thus allowing us to generate a colexification graph of the source language. Combining graphs from more than one language pair may further enrich the colexification graph.

### Code Example
```python
from mapping_of_lexicons.colexifications import get_colexifications, get_colexification_graph, plot_graph

with open(os.path.join(main_dir, file_name, ".json"), "r") as json_file:
    bipartite_graph = json.load(json_file)
words_to_display = ["about", "above", "across", "after", "against", "along", "among", "around", "at", "before",
                       "behind", "below", "beneath", "beside", "between", "beyond", "but", "by", "concerning",
                       "considering", "despite", "down", "during", "except", "for", "from", "in", "inside", "into",
                       "like", "near", "of", "off", "on", "onto", "out", "outside", "over", "past", "regarding",
                       "round", "since", "through", "throughout", "to", "toward", "under", "underneath", "until",
                       "unto", "up", "upon", "with", "within", "without"]
bipartite_graph = {key: bipartite_graph[key] for key in words_to_display if key in bipartite_graph}

G = get_colexification_graph(bipartite_graph)
plot_graph(G)
```

This code yields the following colexification graph, which captures the polysemies between English and Spanish (note the connection between "for" and "by" which derives from figure 1):

![colexification graph](https://github.com/user-attachments/assets/6026c9d7-7ffb-4b37-a86a-c640232499f3)

# References
```bibtex
@article{ilani-et-al-2024,
  title={Assessing the Role of Lexical Semantics in Cross-lingual Transfer through Controlled Manipulations},
  author={Ilani, Roy and Karidi, Taelin and Abend, Omri},
  journal={arXiv preprint arXiv:2408.07599v1},
  year={2024}
}
```
