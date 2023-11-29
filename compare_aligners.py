import json
import os

from conllu import parse
from simalign import simalign

from transformations.semantic_transformation import filter_only_1_to_1_alignment, get_word_alignments, \
    transform_sentences_semantic


def read_conllu_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    sentences = [[token["form"] for token in tree] for tree in parse(data)]
    return sentences


def get_all_possible_sentence_alignments(len_source, len_target):
    return [(i, j) for i in range(len_source) for j in range(len_target)]


def get_alignments_as_tuples(alignments_dic):
    alignments_as_tuples = []
    for key, values in alignments_dic.items():
        for value in values:
            if key != "X" and value != "X":
                alignments_as_tuples.append((int(key)-1, int(value)-1))
    return alignments_as_tuples


en_sentences = read_conllu_sentences(r"data\UD_data\en_pud-ud-test.conllu")
es_sentences = read_conllu_sentences(r"data\UD_data\es_pud-ud-test.conllu")
all_possible_alignments = [get_all_possible_sentence_alignments(len(en_sentence), len(es_sentence))
                           for en_sentence, es_sentence in zip(en_sentences, es_sentences)]

f = open('pud_data.json', encoding="utf8")
en_es_data = json.load(f)["en-es"]
gold_alignments = [get_alignments_as_tuples(i["alignment"]) for i in en_es_data]
gold_alignments = [filter_only_1_to_1_alignment(alignment) for alignment in gold_alignments]

aligner = simalign.SentenceAligner(model="xlmr", token_type="bpe", matching_methods="a", device="cuda:0")
simalign_alignments = [aligner.get_word_aligns(source_words, target_words)['inter']
                       for source_words, target_words in zip(en_sentences, es_sentences)]
simalign_alignments = [filter_only_1_to_1_alignment(alignment) for alignment in simalign_alignments]

with open(os.path.join(r"data\alignments\en-es_alignments", "en-es_alignments_only1to1_clean_inverse_clean_inverse.json"),
          "r") as json_file:
    word_dic = json.load(json_file)
en_sentences_joined = [" ".join(sentence) for sentence in en_sentences]
es_sentences_joined = [" ".join(sentence) for sentence in es_sentences]
_, my_alignments = transform_sentences_semantic(en_sentences_joined, es_sentences_joined, source_language="en",
                                                target_language="es", word_dic=word_dic, no_lemmas=True)


print("done")