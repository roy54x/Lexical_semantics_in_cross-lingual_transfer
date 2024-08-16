import json
import os
import re
import time
import operator
from collections import Counter

import pandas as pd
import scipy
from trankit import Pipeline
import jieba
import simalign

from mapping_of_lexicons.create_graph import filter_only_1_to_1_alignment
from mapping_of_lexicons.tokenize_sentences import light_tokenize_sentences, tokenize_sentences
from utils import transform_all_gts, post_process_sentence


def transform_sentences_semantic(source_sentences, target_sentences, source_language, target_language,
                                 word_dic, lemmas_dic=None, no_lemmas=False, filter_func=None, aligner=None, pipeline=None):

    if no_lemmas:
        if filter_func:
            word_dic = filter_func(word_dic)
        lemmas_dic = word_dic
        source_tokenized_sentences = light_tokenize_sentences(source_sentences, source_language)
        target_tokenized_sentences = light_tokenize_sentences(target_sentences, target_language)
    else:
        if filter_func:
            lemmas_dic = filter_func(lemmas_dic)
        source_tokenized_sentences = tokenize_sentences(source_sentences, source_language, pipeline=pipeline)
        target_tokenized_sentences = tokenize_sentences(target_sentences, target_language, pipeline=pipeline)

    if aligner is None:
        aligner = simalign.SentenceAligner(model="xlmr", token_type="bpe", matching_methods="a", device="cuda:0")
    transformed_sentences = []
    for sentence_idx, (source_sentence, target_sentence) in enumerate(
            zip(source_tokenized_sentences, target_tokenized_sentences)):
        print(sentence_idx, source_sentences[sentence_idx])
        print(sentence_idx, target_sentences[sentence_idx])
        try:
            if no_lemmas:
                source_words, source_lemmas = source_sentence, source_sentence
                target_words, target_lemmas = target_sentence, target_sentence
            else:
                source_words, source_lemmas = [x[0] for x in source_sentence], [x[1] for x in source_sentence]
                target_words, target_lemmas = [x[0] for x in target_sentence], [x[1] for x in target_sentence]
            sentence_alignment = aligner.get_word_aligns(source_words, target_words)['inter']
            sentence_alignment = filter_only_1_to_1_alignment(sentence_alignment, check2to1=False)

            transformed_sentence = []
            for token_idx, (source_word, source_lemma) in enumerate(zip(source_words, source_lemmas)):

                is_word_in_dic = source_lemma in lemmas_dic.keys()
                if is_word_in_dic:
                    aligned_indices = [alignment[1] for alignment in sentence_alignment if alignment[0] == token_idx]
                    aligned_lemmas_and_words = {target_lemmas[aligned_idx]: target_words[aligned_idx] for aligned_idx in
                                                aligned_indices}
                    target_lemmas_dic = lemmas_dic[source_lemma]
                    target_lemmas_aligned_dic = {key: value for key, value in
                                                 target_lemmas_dic.items() if key in aligned_lemmas_and_words.keys()}
                    target_lemmas_exist_dic = {key: value for key, value in
                                               target_lemmas_dic.items() if key in target_lemmas}

                    # option 1: word is aligned to a known lemma in the dictionary
                    if target_lemmas_aligned_dic:
                        best_lemma = max(target_lemmas_aligned_dic.items(), key=operator.itemgetter(1))[0]
                        best_alignment = aligned_lemmas_and_words[best_lemma]

                    # option 2: a lemma in the dictionary is found in the target sentence and there is no better
                    # alignment to this lemma out of all the source lemmas
                    else:
                        is_best_source_to_align_to_this_lemma = False
                        is_lemma_not_aligned = False
                        if target_lemmas_exist_dic:
                            best_lemma = max(target_lemmas_exist_dic.items(), key=operator.itemgetter(1))[0]
                            is_best_source_to_align_to_this_lemma = max({key: lemmas_dic[key][best_lemma] for key in source_lemmas
                                                 if key in lemmas_dic.keys() and best_lemma in lemmas_dic[key].keys()}.items(),
                                                                        key=operator.itemgetter(1))[0] == source_lemma
                            is_lemma_not_aligned = target_lemmas.index(best_lemma) not in [alignment[1] for alignment in sentence_alignment]
                        if is_best_source_to_align_to_this_lemma and is_lemma_not_aligned:
                            best_alignment = target_words[target_lemmas.index(best_lemma)]

                        # option 3: no lemma is found in the target sentence, and therefore
                        # we pick the most common aligned word in the "word_dic"
                        else:
                            if source_word in word_dic.keys():
                                target_word_dic = word_dic[source_word]
                                max_aligned_value = max(list(target_word_dic.values()))
                                max_aligned_index = list(target_word_dic.values()).index(max_aligned_value)
                                best_alignment = list(target_word_dic.keys())[max_aligned_index]

                            # option 4: the lemma or the word does not exist in the dictionary
                            else:
                                best_alignment = source_word
                else:
                    best_alignment = source_word

                transformed_sentence.append(best_alignment)
            transformed_sentence = " ".join(transformed_sentence)

        except Exception as e:
            print(e)
            transformed_sentence = source_sentences[sentence_idx]

        transformed_sentence = post_process_sentence(post_process_sentence(transformed_sentence))
        print(sentence_idx, transformed_sentence)
        print()
        transformed_sentences.append(transformed_sentence)
    return transformed_sentences


if __name__ == '__main__':
    #get_database_alignments(main_dir=r"C:\Users\RoyIlani\pythonProject\data\UM-Corpus", source_language="english", target_language="chinese",
    #                        source_doc_name="en-zh_cn.txt\TED2020.en-zh_cn.en", target_doc_name="en-zh_cn.txt\TED2020.en-zh_cn.zh_cn",
    #                        output_file_name="UM_en-zh_cn_alignments_lemmas_only1to1_2", align_lemmas=True, only_1_to_1_alignment=True,
    #                        iter_size=10000)

    main_dir = r"C:\Users\RoyIlani\pythonProject\data\alignments\en-zh_cn_alignments"
    #save_concat_dic(main_dir, "UM_en-zh_cn_alignments_only1to1",
    #                "TED_en-zh_cn_alignments_only1to1", "en-zh_cn_alignments_only1to1")

    #save_clean_dic(main_dir, "en-zh_cn_alignments_only1to1", 4.0, 0.02)
    #save_inverse_dic(main_dir, "en-zh_cn_alignments_only1to1_clean")
    #save_clean_dic(main_dir, "en-zh_cn_alignments_only1to1_clean_inverse", 4.0, 0.02)
    #save_inverse_dic(main_dir, "en-zh_cn_alignments_only1to1_clean_inverse_clean")

    #get_entropies(main_dir, "en-zh_cn_alignments_only1to1_clean_inverse_clean")
    #get_entropies(main_dir, "en-zh_cn_alignments_only1to1_clean_inverse_clean", to_json=False)
    #get_entropies(main_dir, "en-zh_cn_alignments_only1to1_clean_inverse_clean_inverse")
    #get_entropies(main_dir, "en-zh_cn_alignments_only1to1_clean_inverse_clean_inverse", to_json=False)
    #save_symmetry_group_dic(main_dir, "en-zh_cn_alignments_lemmas_only1to1_clean_inverse_clean_inverse")

    with open(
            os.path.join(main_dir, "en-zh_cn_alignments_lemmas_only1to1_clean_inverse_clean_inverse_0.0015.json"),
            "r") as json_file:
        lemmas_dic = json.load(json_file)
    with open(os.path.join(main_dir, "en-zh_cn_alignments_only1to1_clean_inverse_clean_inverse_0.0015.json"),
              "r") as json_file:
        word_dic = json.load(json_file)

    #entropies_df = pd.read_json(
    #    os.path.join(main_dir, "en-zh_cn_alignments_lemmas_only1to1_clean_inverse_clean_inverse_0.0015_entropies.json"))
    entropies_df = pd.read_json(
        os.path.join(r"data/alignments/en-zh_cn_alignments", "en-zh_cn_alignments_lemmas_only1to1_clean_inverse_clean_0.0015_entropies.json"))
    entropies = {word: entropy for word, entropy in
                 zip(entropies_df["words"].to_list(), entropies_df["entropies"].to_list())}


    def filter_source_entropy(word_dic):
        filtered_dic = filter_by_entropy(word_dic, entropies, min_entropy=1.82, max_entropy=100.00)
        return filtered_dic


    def filter_target_entropy(word_dic):
        inverse_dic = get_inverse_dic(word_dic)
        filtered_inverse_dic = filter_by_entropy(inverse_dic, entropies, min_entropy=1.235, max_entropy=100.0)
        return get_inverse_dic(filtered_inverse_dic)


    transform_all_gts(r"C:\Users\RoyIlani\pythonProject\data\TED_data\TED_en-zh_cn_gt\train_set",
                      r"C:\Users\RoyIlani\pythonProject\data\TED_data\TED_en-zh_cn_gt\train_set",
                      transform_sentences_semantic, "zh_cn_semantic_transformation_from1.235entropy-in-chinese",
                      input_source_col_name="sentence", input_target_col_name="target_sentence", override=False,
                      source_language="english", target_language="chinese",
                     word_dic=word_dic, lemmas_dic=lemmas_dic, no_lemmas=False, filter_func=filter_target_entropy)

    transform_all_gts(r"C:\Users\RoyIlani\pythonProject\data\TED_data\TED_en-zh_cn_gt\test_set",
                      r"C:\Users\RoyIlani\pythonProject\data\TED_data\TED_en-zh_cn_gt\test_set",
                      transform_sentences_semantic, "zh_cn_semantic_transformation_from1.235entropy-in-chinese",
                      input_source_col_name="sentence", input_target_col_name="target_sentence", override=False,
                      source_language="english", target_language="chinese",
                      word_dic=word_dic, lemmas_dic=lemmas_dic, no_lemmas=False, filter_func=filter_target_entropy)