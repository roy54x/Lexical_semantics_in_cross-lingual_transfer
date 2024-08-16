import json
import os

import scipy
import pandas as pd


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
