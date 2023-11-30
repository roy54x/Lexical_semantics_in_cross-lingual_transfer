import json
import os

from conllu import parse
from simalign import simalign

from transformations.semantic_transformation import filter_only_1_to_1_alignment, transform_sentences_semantic


def read_conllu_sentences(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    sentences = [[token["form"] for token in tree] for tree in parse(data)]
    return sentences


def get_alignments_as_tuples(alignments_dic):
    alignments_as_tuples = []
    for key, values in alignments_dic.items():
        for value in values:
            if key != "X" and value != "X":
                alignments_as_tuples.append((int(key)-1, int(value)-1))
    return alignments_as_tuples


def calculate_f1_score(gold_alignments, predicted_alignments):
    true_positive = 0
    false_positive = 0
    false_negative = 0

    for gold_sentence_alignments, predicted_sentence_alignments in zip(gold_alignments, predicted_alignments):
        true_positive += len(set(gold_sentence_alignments) & set(predicted_sentence_alignments))
        false_positive += len(set(predicted_sentence_alignments) - set(gold_sentence_alignments))
        false_negative += len(set(gold_sentence_alignments) - set(predicted_sentence_alignments))

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0

    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    return f1, true_positive, false_positive, false_negative


en_sentences = read_conllu_sentences(r"C:\Users\RoyIlani\pythonProject\data\UD_data\en_pud-ud-test.conllu")
es_sentences = read_conllu_sentences(r"C:\Users\RoyIlani\pythonProject\data\UD_data\es_pud-ud-test.conllu")

f = open('pud_data.json', encoding="utf8")
en_es_data = json.load(f)["en-es"]
gold_alignments = [get_alignments_as_tuples(i["alignment"]) for i in en_es_data]
gold_alignments = [filter_only_1_to_1_alignment(alignment) for alignment in gold_alignments]

aligner = simalign.SentenceAligner(model="xlmr", token_type="bpe", matching_methods="a", device="cuda:0")
simalign_alignments = [aligner.get_word_aligns(source_words, target_words)['inter']
                       for source_words, target_words in zip(en_sentences, es_sentences)]
simalign_alignments = [filter_only_1_to_1_alignment(alignment) for alignment in simalign_alignments]

with open(os.path.join(r"C:\Users\RoyIlani\pythonProject\data\alignments\en-es_alignments", "en-es_alignments_only1to1_clean_inverse_clean_inverse.json"),
          "r") as json_file:
    word_dic = json.load(json_file)
en_sentences_joined = [" ".join(sentence) for sentence in en_sentences]
es_sentences_joined = [" ".join(sentence) for sentence in es_sentences]
_, my_alignments = transform_sentences_semantic(en_sentences_joined, es_sentences_joined, source_language="en",
                                                target_language="es", word_dic=word_dic, no_lemmas=True)

simalign_f1, simalign_true_positive, simalign_false_positive, simalign_false_negative = calculate_f1_score(gold_alignments, simalign_alignments)
my_f1, my_true_positive, my_false_positive, my_false_negative = calculate_f1_score(gold_alignments, my_alignments)

print("simalign F1 score is: " + str(simalign_f1))
print("my F1 score is: " + str(my_f1))