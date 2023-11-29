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

from utils import load_doc, parse_sentences_from_doc, transform_all_gts, post_process_sentence


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
    processed_alignments = []
    for sentence_idx, (source_sentence, target_sentence) in enumerate(
            zip(source_tokenized_sentences, target_tokenized_sentences)):
        print(sentence_idx, source_sentences[sentence_idx])
        print(sentence_idx, target_sentences[sentence_idx])

        if no_lemmas:
            source_words, source_lemmas = source_sentence, source_sentence
            target_words, target_lemmas = target_sentence, target_sentence
        else:
            source_words, source_lemmas = [x[0] for x in source_sentence], [x[1] for x in source_sentence]
            target_words, target_lemmas = [x[0] for x in target_sentence], [x[1] for x in target_sentence]
        sentence_alignment = aligner.get_word_aligns(source_words, target_words)['inter']
        sentence_alignment = filter_only_1_to_1_alignment(sentence_alignment, check2to1=False)

        try:
            transformed_sentence = []
            processed_sentence_alignment = []
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
                        processed_sentence_alignment.append((token_idx, target_words.index(best_alignment)))

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
                            processed_sentence_alignment.append((token_idx, target_words.index(best_alignment)))

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
            processed_sentence_alignment = sentence_alignment

        transformed_sentence = post_process_sentence(post_process_sentence(transformed_sentence))
        print(sentence_idx, transformed_sentence)
        print()
        transformed_sentences.append(transformed_sentence)
        processed_alignments.append(processed_sentence_alignment)
    return transformed_sentences, processed_alignments


def tokenize_sentences(sentences, language, pipeline=None):
    if not pipeline:
        pipeline = Pipeline(language, gpu=True)
    else:
        pipeline.set_active(language)

    tokenized_sentences = []
    for sentence in sentences:
        try:
            if not sentence:
                raise Exception
            if language == "chinese":
                tokenized_sentence = [token_tuple[0] for token_tuple in jieba.tokenize(sentence)]
                tokenized_sentence = list(filter(lambda token: token != "" and not re.match("\s", token), tokenized_sentence))
                tokenized_sentence = [(token, token) for token in tokenized_sentence]
            else:
                tokenized_sentence = []
                lemmatized_sentence = pipeline.lemmatize(sentence, is_sent=True)["tokens"]
                for token in lemmatized_sentence:
                    if isinstance(token["id"], int):
                        tokenized_sentence.append((token["text"], token["lemma"]))
                    else:
                        tokenized_sentence.extend([(sub_token["text"], sub_token["lemma"])
                                           for sub_token in token['expanded']])
        except Exception:
            tokenized_sentences.append(None)
            print("Error tokenizing sentence")
            print()
            continue
        tokenized_sentences.append(tokenized_sentence)
    return tokenized_sentences


def light_tokenize_sentences(sentences, language):
    tokenized_sentences = []
    for sentence in sentences:
        if not sentence:
            tokenized_sentences.append(None)
            print("Error tokenizing sentence")
            print()
            continue
        if language == "chinese":
            tokenized_sentence = [token_tuple[0] for token_tuple in jieba.tokenize(sentence)]
        else:
            tokenized_sentence = sentence.lower()
            tokenized_sentence = tokenized_sentence.replace(".", " .")
            tokenized_sentence = tokenized_sentence.replace(",", " ,")
            tokenized_sentence = tokenized_sentence.replace("?", " ?")
            tokenized_sentence = tokenized_sentence.replace(";", " ;")
            tokenized_sentence = tokenized_sentence.split(" ")
        tokenized_sentence = list(filter(lambda token: token != "" and not re.match("\s", token), tokenized_sentence))
        tokenized_sentences.append(tokenized_sentence)
    return tokenized_sentences


def get_word_alignments(source_sentences, target_sentences, word_dic, alignments_file=None,
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


def get_database_alignments(main_dir=r"data/europarl_data", source_language="english", target_language="spanish",
                            source_doc_name="europarl-v7.es-en.en", target_doc_name="europarl-v7.es-en.es",
                            output_file_name="europarl_en-es_alignments", align_lemmas=True, only_1_to_1_alignment=True,
                            iter_size=100000):
    first_time = time.time()
    source_doc = load_doc(os.path.join(main_dir, source_doc_name))
    target_doc = load_doc(os.path.join(main_dir, target_doc_name))
    source_sentences = parse_sentences_from_doc(source_doc)
    target_sentences = parse_sentences_from_doc(target_doc)
    print("Data ready. time taken: " + str(round(time.time() - first_time, 3)))

    word_dic = {}
    for i in range(len(source_sentences) // iter_size):
        second_time = time.time()
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

        third_time = time.time()
        print("Tokenization done, iteration number: " + str(i) + ". time taken: " + str(
            round(third_time - second_time, 3)))

        alignments_file = open(os.path.join(main_dir, output_file_name + ".txt"), "a")
        word_dic = get_word_alignments(source_tokenized_sentences, target_tokenized_sentences,
                                       word_dic, alignments_file, align_lemmas=align_lemmas,
                                       only_1_to_1_alignment=only_1_to_1_alignment)
        with open(os.path.join(main_dir, output_file_name + ".json"), "w") as outfile:
            json.dump(word_dic, outfile)
        alignments_file.close()
        print("Alignment done, iteration number: " + str(i) + ". time taken: "
              + str(round(time.time() - third_time, 3)))


def save_inverse_dic(main_dir, file_name):
    file_path = os.path.join(main_dir, file_name + ".json")
    with open(file_path, "r") as json_file:
        dic = json.load(json_file)
    inverse_dic = get_inverse_dic(dic)
    output_path = os.path.join(main_dir, file_name + "_inverse.json")
    with open(output_path, "w") as outfile:
        json.dump(inverse_dic, outfile)


def get_inverse_dic(dic):
    inverse_dic = {}
    for (source_key, target_val) in dic.items():
        for (target_key, alignment_val) in target_val.items():
            if target_key not in inverse_dic.keys():
                inverse_dic[target_key] = {source_key: alignment_val}
            else:
                inverse_dic[target_key][source_key] = alignment_val
    return inverse_dic


def save_clean_dic(main_dir, file_name, min_amount, min_precent):
    file_path = os.path.join(main_dir, file_name + ".json")
    with open(file_path, "r") as json_file:
        dic = json.load(json_file)
    clean_dic = {}
    for (source_key, target_val) in dic.items():
        alignments_sum = sum(target_val.values())
        min_val = max(alignments_sum * min_precent, min_amount)
        clean_target_val = {target_key: alignment_val for (target_key, alignment_val)
                            in target_val.items() if alignment_val > min_val}
        if clean_target_val:
            clean_dic[source_key] = clean_target_val
    output_path = os.path.join(main_dir, file_name + "_clean.json")
    with open(output_path, "w") as outfile:
        json.dump(clean_dic, outfile)


def save_concat_dic(main_dir, file_name_1, file_name_2, output_name):
    file_path = os.path.join(main_dir, file_name_1 + ".json")
    with open(file_path, "r") as json_file:
        dic_1 = json.load(json_file)
    file_path = os.path.join(main_dir, file_name_2 + ".json")
    with open(file_path, "r") as json_file:
        dic_2 = json.load(json_file)
    concat_dic = dic_1
    for (source_key, target_val) in dic_2.items():
        if source_key in concat_dic.keys():
            for (target_key, alignment_val) in target_val.items():
                if target_key in concat_dic[source_key].keys():
                    concat_dic[source_key][target_key] += alignment_val
                else:
                    concat_dic[source_key][target_key] = alignment_val
        else:
            concat_dic[source_key] = target_val
    output_path = os.path.join(main_dir, output_name + ".json")
    with open(output_path, "w") as outfile:
        json.dump(concat_dic, outfile)


def save_symmetry_group_dic(main_dir, file_name):
    file_path = os.path.join(main_dir, file_name + ".json")
    with open(file_path, "r") as json_file:
        dic = json.load(json_file)
    inverse_dic = get_inverse_dic(dic)

    all_symmetry_groups_dic = {}
    source_keys, target_keys = set(dic.keys()), set(inverse_dic.keys())
    group_idx = 0
    while source_keys:
        group_source_keys = {next(iter(source_keys))}
        group_last_size = 0
        while len(group_source_keys) != group_last_size:
            group_last_size = len(group_source_keys)
            for source_key in group_source_keys:
                if source_key not in source_keys:
                    continue
                new_group_target_keys = set(dic[source_key].keys())
                for target_key in new_group_target_keys:
                    if target_key not in target_keys:
                        continue
                    new_group_source_keys = set(inverse_dic[target_key].keys())
                    group_source_keys = group_source_keys.union(new_group_source_keys)
                target_keys = target_keys.difference(new_group_target_keys)
                source_keys.remove(source_key)
        group_dic = {key: dic[key] for key in group_source_keys}
        all_symmetry_groups_dic["symmetry_group_number_" + str(group_idx)] = group_dic
        group_idx += 1

    output_path = os.path.join(main_dir, file_name + "_symmetry-groups.json")
    with open(output_path, "w") as outfile:
        json.dump(all_symmetry_groups_dic, outfile)


def get_entropies(main_dir, file_name, to_json=True):
    file_path = os.path.join(main_dir, file_name + ".json")
    with open(file_path, "r") as json_file:
        dic = json.load(json_file)
    words = dic.keys()
    amounts, entropies = [], []
    for word in words:
        appearances = dic[word].values()
        amounts.append(sum(appearances))
        entropies.append(round(scipy.stats.entropy(list(appearances)), 3))
    df = pd.DataFrame({"words": words, "amounts": amounts, "entropies": entropies})
    if to_json:
        output_path = os.path.join(main_dir, file_name + "_entropies.json")
        df.to_json(output_path)
    else:
        output_path = os.path.join(main_dir, file_name + "_entropies.csv")
        df.to_csv(output_path)


def filter_out_1to2(word_dic):
    filtered_word_dic = {key: value for key, value in word_dic.items() if len(list(value.keys())) < 2}
    return filtered_word_dic


def filter_out_2to1(word_dic):
    inverse_dic = get_inverse_dic(word_dic)
    filtered_inverse_dic = filter_out_1to2(inverse_dic)
    filtered_word_dic = get_inverse_dic(filtered_inverse_dic)
    return filtered_word_dic


def filter_only_1to1(word_dic):
    filtered_word_dic = filter_out_2to1(filter_out_1to2(word_dic))
    return filtered_word_dic


def filter_by_entropy(word_dic, entropies, min_entropy=0.0, max_entropy=100.0):
    filtered_word_dic = {key: value for key, value in word_dic.items() if
                         max_entropy > entropies[key] >= min_entropy}
    return filtered_word_dic


def filter_all_words_1to1(word_dic):
    new_dic = {}
    new_dic_values = []
    for source_key, source_value in word_dic.items():
        source_value_filtered = {key: value for key, value in source_value.items() if key not in new_dic_values}
        target_values_list = list(source_value_filtered.values())
        if not target_values_list:
            continue
        max_target_value = max(target_values_list)
        max_index = target_values_list.index(max_target_value)
        max_target_key = list(source_value_filtered.keys())[max_index]
        new_dic[source_key] = {max_target_key: max_target_value}
        new_dic_values.append(max_target_key)
    return new_dic


def filter_only_synonyms(word_dic):
    inverse_dic = get_inverse_dic(word_dic)
    source_keys = set(word_dic.keys())
    new_dic = {}
    while source_keys:
        source_key = next(iter(source_keys))
        aligned_source_keys = {source_key}
        for target_key in word_dic[source_key].keys():
            aligned_source_keys.update(set(inverse_dic[target_key].keys()))
        aligned_source_keys = aligned_source_keys.intersection(source_keys, aligned_source_keys)
        aligned_source_keys_dic = {key: sum(word_dic[key].values()) for key in aligned_source_keys}
        max_aligned_source_key_and_value = max(aligned_source_keys_dic.items(), key=operator.itemgetter(1))
        for aligned_source_key in aligned_source_keys:
            new_dic[aligned_source_key] = {max_aligned_source_key_and_value[0]: max_aligned_source_key_and_value[1]}
            source_keys.remove(aligned_source_key)
    return new_dic


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