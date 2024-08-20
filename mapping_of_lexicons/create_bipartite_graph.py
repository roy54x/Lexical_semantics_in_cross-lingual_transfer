import json
import os
import time
from collections import Counter

import simalign

from mapping_of_lexicons.tokenize_sentences import tokenize_sentences, light_tokenize_sentences


def create_bipartite_graph(source_sentences, target_sentences, source_language="english", target_language="spanish",
                           output_file_name="europarl_en-es_alignments", align_lemmas=True, only_1_to_1_alignment=True,
                           iter_size=100000):
    word_dic = {}
    for i in range(len(source_sentences) // iter_size):
        tokenization_time = time.time()
        print("starting iteration number: " + str(i))

        if align_lemmas:
            source_tokenized_sentences = tokenize_sentences(source_sentences[i * iter_size:(i + 1) * iter_size],
                                                            source_language)
            target_tokenized_sentences = tokenize_sentences(target_sentences[i * iter_size:(i + 1) * iter_size],
                                                            target_language)
        else:
            source_tokenized_sentences = light_tokenize_sentences(source_sentences[i * iter_size:(i + 1) * iter_size],
                                                                  source_language)
            target_tokenized_sentences = light_tokenize_sentences(target_sentences[i * iter_size:(i + 1) * iter_size],
                                                                  target_language)

        graph_creation_time = time.time()
        print("Tokenization done, iteration number: " + str(i) + ". time taken: " + str(
            round(graph_creation_time - tokenization_time, 3)))

        alignments_file = open(os.path.join(main_dir, output_file_name + ".txt"), "a")
        word_dic = _create_graph_from_tokenized_sentences(source_tokenized_sentences, target_tokenized_sentences,
                                                          word_dic, alignments_file, align_lemmas=align_lemmas,
                                                          only_1_to_1_alignment=only_1_to_1_alignment)
        with open(os.path.join(main_dir, output_file_name + ".json"), "w") as outfile:
            json.dump(word_dic, outfile)
        alignments_file.close()
        print("Alignment done, iteration number: " + str(i) + ". time taken: "
              + str(round(time.time() - graph_creation_time, 3)))


def _create_graph_from_tokenized_sentences(source_sentences, target_sentences, word_dic, alignments_file=None,
                                           align_lemmas=True, only_1_to_1_alignment=True):
    aligner = simalign.SentenceAligner(model="xlmr", token_type="bpe", matching_methods="a", device="cuda:0")
    for (source_sentence, target_sentence) in zip(source_sentences, target_sentences):
        if not source_sentence or not target_sentence:
            if alignments_file:
                alignments_file.write("\n")
            print("Error aligning sentence")
            print()
            continue

        if not align_lemmas:
            source_words = source_sentence
            target_words = target_sentence
        else:
            source_words, source_lemmas = [x[0] for x in source_sentence], [x[1] for x in source_sentence]
            target_words, target_lemmas = [x[0] for x in target_sentence], [x[1] for x in target_sentence]
        sentence_alignment = aligner.get_word_aligns(source_words, target_words)['inter']
        if only_1_to_1_alignment:
            sentence_alignment = filter_only_1_to_1_alignment(sentence_alignment)
        print(source_words)
        print(target_words)
        print(sentence_alignment)
        print()

        if alignments_file:
            alignments_file.write(str(sentence_alignment) + "\n")
        for alignment in sentence_alignment:
            if align_lemmas:
                source_word = source_lemmas[alignment[0]]
                target_word = target_lemmas[alignment[1]]
            else:
                source_word = source_words[alignment[0]]
                target_word = target_words[alignment[1]]
            if source_word not in word_dic.keys():
                word_dic[source_word] = {target_word: 1.0}
            elif target_word not in word_dic[source_word].keys():
                word_dic[source_word][target_word] = 1.0
            else:
                word_dic[source_word][target_word] += 1.0
    return word_dic


def filter_only_1_to_1_alignment(sentence_alignment, check1to2=True, check2to1=True):
    invalid_x, invalid_y = [], []
    if check2to1:
        invalid_x = [key for key, value in Counter([alignment[0] for alignment in sentence_alignment]).items() if
                     value > 1]
    if check1to2:
        invalid_y = [key for key, value in Counter([alignment[1] for alignment in sentence_alignment]).items() if
                     value > 1]
    return [(x, y) for x, y in sentence_alignment if (x not in invalid_x and y not in invalid_y)]


if __name__ == '__main__':
    from utils.utils import load_doc, parse_sentences_from_doc

    main_dir = r"data/europarl_data"
    source_doc_name, target_doc_name = "europarl-v7.es-en.en", "europarl-v7.es-en.es"
    parsing_time = time.time()
    source_doc = load_doc(os.path.join(main_dir, source_doc_name))
    target_doc = load_doc(os.path.join(main_dir, target_doc_name))
    source_sentences = parse_sentences_from_doc(source_doc)
    target_sentences = parse_sentences_from_doc(target_doc)
    print("Data ready. time taken: " + str(round(time.time() - parsing_time, 3)))

    create_bipartite_graph(source_sentences, target_sentences, source_language="english", target_language="spanish",
                           output_file_name="europarl_en-es_alignments")
