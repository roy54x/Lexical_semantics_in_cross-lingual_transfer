# Lexical Semantics in Cross-Lingual Transfer

This repository provides the code to map lexicons of two languages, as discussed in [Assessing the Role of Lexical Semantics in Cross-lingual Transfer through Controlled Manipulations](https://arxiv.org/abs/2408.07599v1). The main functionalities include extracting bipartite graphs, calculating translation entropy, and generating colexification graphs.

## Table of Contents
1. [Installation](#installation)
2. [Download bitext](#Download-bitext)
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

## Download bitext

Download a parallel corpus for the two languages you wish to map their lexicons. For example, if you want to map two European languages, you can use the Europarl corpus available at https://www.statmt.org/europarl/.

## Extracting the Bipartite Graph
