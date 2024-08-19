import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from utils.clean_data import clean_parallel_data


def get_all_gts(model, data_path, output_path, saving_size=50000):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    existing_files = os.listdir(output_path)
    df = pd.DataFrame({"sentence": [], "input_ids": [], "attention_mask": [], "embedding": []})
    counter = 0
    for file_name in os.listdir(data_path):
        if file_name[:-4]+".npy" in existing_files:
            continue
        file_path = os.path.join(data_path, file_name)
        file_df = get_file_gts(model, file_path)
        df = pd.concat([df, file_df])
        if len(df) >= saving_size:
            df2save = df[:saving_size].reset_index(drop=True)
            output_file_path = os.path.join(output_path, "sentence_gt_"+str(counter)+".json")
            df2save.to_json(output_file_path)
            df = df[saving_size:].reset_index(drop=True)
            counter += 1
        print(file_name)


def get_file_gts(model, file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
        sentences = text.split("\n")
        sentences = [sentence for sentence in sentences if sentence]
        tokenized_sentences = model.tokenizer(sentences, max_length=26, padding='max_length', truncation=True)
        input_ids = [np.array(sentence_tokens) for sentence_tokens in tokenized_sentences["input_ids"]]
        attention_masks = [np.array(sentence_tokens) for sentence_tokens in tokenized_sentences["attention_mask"]]
        embeddings = [sentence_embedding for sentence_embedding in model.encode(sentences)]
    file.close()
    return pd.DataFrame({"sentence": sentences, "input_ids": input_ids, "attention_mask": attention_masks, "embedding": embeddings})


def save_parallel_gts(model, parent_main_dir, source_path, target_path, source_language,
                      target_language, output_path, saving_size=50000):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    filtered_source_sentences, filtered_target_sentences = clean_parallel_data(parent_main_dir, source_path, target_path,
                                                                               source_language=source_language, target_language=target_language)
    tokenized_sentences = model.tokenizer(filtered_source_sentences, max_length=26, padding='max_length', truncation=True)
    input_ids = [np.array(sentence_tokens) for sentence_tokens in tokenized_sentences["input_ids"]]
    attention_masks = [np.array(sentence_tokens) for sentence_tokens in tokenized_sentences["attention_mask"]]
    embeddings = [sentence_embedding for sentence_embedding in model.encode(filtered_source_sentences)]
    df = pd.DataFrame({"sentence": filtered_source_sentences, "target_sentence": filtered_target_sentences,
                       "input_ids": input_ids, "attention_mask": attention_masks, "embedding": embeddings})
    for counter in range(len(df)//saving_size):
        df2save = df[counter*saving_size:(counter+1)*saving_size].reset_index(drop=True)
        output_file_path = os.path.join(output_path, "sentence_gt_" + str(counter) + ".json")
        df2save.to_json(output_file_path)


if __name__ == '__main__':
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    #data_path = os.path.join("data", "cc100_data_clean","cc100_en")
    #output_path = os.path.join("data", "cc100_data_clean", "cc100_en_gt")
    #get_all_gts(model, data_path, output_path)

    parent_main_dir = r"C:\Users\RoyIlani\pythonProject\data\TED_data\en-he.txt"
    source_path, target_path = "TED2020.en-he.en", "TED2020.en-he.he"
    output_path = os.path.join("data", "TED_data", "TED_en-he_gt")
    save_parallel_gts(model, parent_main_dir, source_path, target_path,
                      source_language="en", target_language="he", output_path=output_path, saving_size=10000)