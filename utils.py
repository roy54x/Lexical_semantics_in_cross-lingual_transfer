import os
import re

import pandas as pd


def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    text = file.read()
    file.close()
    return text


def parse_sentences_from_doc(doc):
    sentences = doc.strip().split('\n')
    parsed_sentences = []
    for sentence in sentences:
        sentence = re.sub('([-.,/!?()])', r' \1 ', sentence)
        sentence = re.sub('\s{2,}', ' ', sentence)
        parsed_sentences.append(sentence.lower())
    return parsed_sentences


def transform_all_files(input_dir, output_dir, transform_func, **kwargs):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    existing_files = os.listdir(output_dir)
    for file_name in os.listdir(input_dir):
        if file_name in existing_files:
            continue
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            file_sentences = text.split("\n")
            file_sentences = transform_func(file_sentences, **kwargs)
        output_file_path = os.path.join(output_dir, file_name)
        with open(output_file_path, 'w', encoding='utf-8') as fp:
            fp.write('\n'.join(file_sentences))
        print(file_name)


def transform_all_gts(input_dir, output_dir, transform_func, output_col_name, override=False, **kwargs):
    for file_name in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file_name)
        file_df = pd.read_json(file_path)
        if output_col_name in file_df.columns and not override:
            continue
        if len(kwargs) > 1:
            file_df[output_col_name] = transform_func(file_df[kwargs["input_source_col_name"]].to_list(),
                                                      file_df[kwargs["input_target_col_name"]].to_list(),
                                                      source_language=kwargs["source_language"],
                                                      target_language=kwargs["target_language"],
                                                      word_dic=kwargs["word_dic"],
                                                      lemmas_dic=kwargs["lemmas_dic"],
                                                      no_lemmas=kwargs["no_lemmas"],
                                                      filter_func=kwargs["filter_func"])
        else:
            file_df[output_col_name] = transform_func(file_df["sentence"].to_list(),
                                                      language=kwargs["language"])
        if output_dir:
            file_output_path = os.path.join(output_dir, file_name)
        else:
            file_output_path = os.path.join(input_dir, file_name)
        file_df.to_json(file_output_path)


def post_process_sentence(sentence):
    sentence = sentence.replace(" ' ' ", " ")
    sentence = sentence.replace(" ' ", " ")
    sentence = sentence.replace("-.", ".")
    sentence = sentence.replace("-?", "?")
    sentence = sentence.replace("-!", "!")
    sentence = sentence.replace(",.", ".")
    sentence = sentence.replace(",!", "!")
    sentence = sentence.replace(",?", "?")
    sentence = sentence.replace(";.", ".")
    sentence = sentence.replace(";!", "!")
    sentence = sentence.replace(";?", "?")
    sentence = sentence.replace(", , ,", ",")
    sentence = sentence.replace(", ,", ",")
    sentence = sentence.replace("; ,", ",")
    sentence = sentence.replace(", ;", ",")
    sentence = sentence.replace("; ;", ";")
    sentence = sentence.replace(" .", ".")
    sentence = sentence.replace(" !", "!")
    sentence = sentence.replace(" ?", "?")
    sentence = sentence.replace(" ,", ",")
    sentence = sentence.replace(" ;", ";")
    return sentence
