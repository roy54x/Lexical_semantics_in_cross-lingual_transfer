# Lexical Semantics in Cross-Lingual Transfer

This repository provides the code to map lexicons of two languages, as discussed in [Assessing the Role of Lexical Semantics in Cross-lingual Transfer through Controlled Manipulations](https://arxiv.org/abs/2408.07599v1). The main functionalities include extracting bipartite graphs, calculating translation entropy, and generating colexification graphs.

## Table of Contents
1. [Installation](#installation)
2. [Download Bitext](#Download-bitext)
3. [Extracting the Bipartite Graph](#extracting-the-bipartite-graph)
4. [Calculating Translation Entropy](#calculating-translation-entropy)
5. [Generating the Colexification Graph](#generating-the-colexification-graph)

## Installation

First, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/roy54x/Lexical_Semantics_in_Cross-Lingual_Transfer.git
cd Lexical_Semantics_in_Cross-Lingual_Transfer
pip install -r requirements.txt
```

## Download Bitext

Download a parallel corpus for the two languages you wish to map their lexicons. For example, if you want to map two European languages, you can use the Europarl corpus available at https://www.statmt.org/europarl/.

## Extracting the Bipartite Graph

\begin{figure}
\begin{tikzpicture}[node distance={5mm}, main/.style = {draw, circle}, every fit/.style={ellipse,draw,inner sep=-2pt,text width=2cm}]

\begin{scope}[start chain=going below,node distance=5mm]
\node[main] (1) {\textit{for}}; 
\node[main] (2) [below=1.0] {\textit{by}}; 
\end{scope}

\begin{scope}[xshift=5cm,yshift=0cm,start chain=going below,node distance=5mm]
\node[main] (3) {\textit{por}};
\node[main] (4) [below=1.0] {\textit{para}};
\end{scope}

\node [fit=(1) (2),label=above:\textit{English}] {};
\node [fit=(3) (4),label=above:\textit{Spanish}] {};

\draw (1) -- node[midway, above, sloped]{85303} (3);
\draw (1) -- node[midway, above left, sloped]{175771} (4);
\draw (2) -- node[midway, below left, sloped]{93781} (3);

\end{tikzpicture} 
\caption{Illustration of the weighted sub-graph which results from the \href{https://www.statmt.org/europarl/}{\textit{Europarl}} parallel corpus. The edges represent the possibility that two words are translations of each other. The weights denote the number of occurrences that each word pair is aligned in the bitext.}\label{fig:weighted_subgraph}
\end{figure}

The bipartite graph represents the mapping between the lexicons of two languages, denoted as $L_s$ (source language) and $L_t$ (target language). Formally, we define a weighted bipartite graph $G = (V_s, V_t, E, w)$, where:

- $V_s$ is the set of words in the lexicon of $L_s$,
- $V_t$ is the set of words in the lexicon of $L_t$,
- $E$ is the set of edges where a pair $(v, u) \in V_s \times V_t$ exists if the word $v$ in $L_s$ is aligned with the word $u$ in $L_t$ in at least one instance in the bitext,
- $w:E \rightarrow \mathbb{N}^+$ is a weight function assigning the number of times each word pair is aligned in the bitext.

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


