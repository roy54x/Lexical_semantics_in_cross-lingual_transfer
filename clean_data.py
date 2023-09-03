import os
import re

from import_data import parse_data
from utils import transform_all_files, load_doc


def clean_sentences(sentences, language="en"):
    filtered_sentences = []
    for sentence in sentences:
        if sentence:
            new_sentences = clean_sentence(sentence, language)
            filtered_sentences.extend(new_sentences)
    return filtered_sentences


def clean_sentence(sentence, language):
    sentence = preprocess_sentence(sentence, language)
    if language == "he":
        new_sentences = [sentence] if sentence.count(".") <= 1 and len(re.findall("[.?!]", sentence[-2:])) == 1  \
                                    and sentence.count(" ") < 16 else []
    elif language == "zh_cn":
        sentence = "。" + sentence
        sentence = sentence.replace("。", "。。").replace("？", "？？").replace("！", "！！")
        new_sentences = re.findall("[。？！]((?:[\s\u4e00-\u9fffa-z0-9%，：·一-]){5,25}[。？！])+", sentence)
    elif language == "el":
        sentence = ". " + sentence.lower() + " "
        sentence = duplicate_regex("[.;?!]\s", sentence)
        new_sentences = re.findall("[.;?!]\s((?:[a-zα-ωίϊΐόάέύϋΰήώ0-9%,'_-]+[,:]?\s){3,15}[a-zα-ωίϊΐόάέύϋΰήώ0-9%,'_-]+[.;?!])\s", sentence)
    else:
        sentence = ". " + sentence.lower() + " "
        sentence = duplicate_regex("[.?!]\s", sentence)
        new_sentences = re.findall("[.?!]\s((?:[a-zñáéíóúü0-9'_-]+[,;:]?\s){3,15}[a-zñáéíóúü0-9'_-]+[.?!])\s", sentence)
    return new_sentences


def preprocess_sentence(sentence, language):
    if not sentence:
        return sentence
    if language == "el":
        sentence = sentence.rstrip()
        if sentence[-1] in [",", ":"]:
            sentence = sentence[:-1] + "."
        elif sentence[-1] not in [".", ";", "!"]:
            sentence += "."
        sentence = sentence.replace("(γέλια) ", "").replace("( Γέλια )", "")
        sentence = sentence.replace("(χειροκρότημα) ", "").replace("( Χειροκρότημα ) ", "")
    elif language == "zh_cn":
        sentence = sentence.rstrip()
        if sentence[-1] in ["，"]:
            sentence = sentence[:-1] + "。"
        elif sentence[-1] not in ["。", "？", "！"]:
            sentence += "。"
        sentence = sentence.replace("（笑声）", "").replace("（笑声。 ） ", "")
        sentence = sentence.replace("（掌声）", "").replace("（掌声。 ） ", "")
    elif language == "he":
        sentence = sentence.replace("(תשואות) ", "").replace("(מחיאות כפיים) ", "")
        sentence = sentence.replace("(צחוק) ", "").replace("(צחוק בקהל) ", "")
    else:
        sentence = sentence.replace("(laughter) ", "").replace("( Laughter ) ", "")
        sentence = sentence.replace("(applause) ", "").replace("( Applause ) ", "")
    return sentence


def duplicate_regex(regex, sentence):
    new_sentence = ""
    for i in range(len(sentence)-1):
        if re.match(regex, sentence[i:i+2]):
            new_sentence += sentence[i:i+2] + sentence[i:i+2]
        else:
            new_sentence += sentence[i:i + 2]
        new_sentence = new_sentence[:-1]
    return new_sentence


def clean_parallel_data(parent_main_dir, source_path, target_path, source_language="en", target_language="es"):
    source_doc = load_doc(os.path.join(parent_main_dir,source_path))
    target_doc = load_doc(os.path.join(parent_main_dir,target_path))
    source_sentences, target_sentences = parse_data(source_doc), parse_data(target_doc)
    filtered_source_sentences, filtered_target_sentences = [], []
    for source_sentence, target_sentence in zip(source_sentences, target_sentences):
        new_source_sentences = clean_sentence(source_sentence, language=source_language)
        new_target_sentences = clean_sentence(target_sentence, language=target_language)
        if new_source_sentences and (len(new_source_sentences) == len(new_target_sentences)):
            filtered_source_sentences.extend(new_source_sentences)
            filtered_target_sentences.extend(new_target_sentences)
    return filtered_source_sentences, filtered_target_sentences


if __name__ == '__main__':
    data_path = os.path.join("data\\cc100_data", "cc100_el")
    output_path = os.path.join("data\\cc100_data_clean", "cc100_el")
    transform_all_files(data_path, output_path, clean_sentences)
