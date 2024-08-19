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

To extract the bipartite graph that maps the two lexicons (see an example in Figure 1), you can follow this code example:

```bash
from utils.utils import load_doc, parse_sentences_from_doc
from mapping_of_lexicons.create_bipartite_graph import create_bipartite_graph

main_dir = r"data/europarl_data"
source_doc_name, target_doc_name = "europarl-v7.es-en.en", "europarl-v7.es-en.es"

source_doc = load_doc(os.path.join(main_dir, source_doc_name))
target_doc = load_doc(os.path.join(main_dir, target_doc_name))
source_sentences = parse_sentences_from_doc(source_doc)
target_sentences = parse_sentences_from_doc(target_doc)

create_bipartite_graph(source_sentences, target_sentences, source_language="english", target_language="spanish",
                           output_file_name="europarl_en-es_alignments")
```


