import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


def get_X_and_y(model, source_df, teacher=True, tokenizer=None, transform_func=None, target_language="spanish", **kwargs):
    X = []
    y = []
    for idx, (_, row) in enumerate(source_df.iterrows()):
        if teacher:
            sentence_1_embedding = np.array(model.encode([row["sentence1"].lower()]))
            sentence_2_embedding = np.array(model.encode([row["sentence2"].lower()]))
        else:
            sentence_pair = [row["sentence1"].lower(), row["sentence2"].lower()]
            if transform_func is not None:
                if len(kwargs) == 0:
                    sentence_pair = transform_func(sentence_pair, target_language)
                else:
                    target_row = kwargs["target_df"].iloc[idx]
                    target_sentence_pair = [target_row["sentence1"].lower(), target_row["sentence2"].lower()]
                    sentence_pair = transform_func(sentence_pair, target_sentence_pair, kwargs["source_language"], target_language,
                                                   word_dic=kwargs["word_dic"], lemmas_dic=kwargs["lemmas_dic"],
                                                   aligner=kwargs["aligner"], pipeline=kwargs["pipeline"])

            tokenized_sentence_1 = tokenizer(sentence_pair[0], max_length=26,
                                            padding='max_length', truncation=True)["input_ids"]
            tokenized_sentence_2 = tokenizer(sentence_pair[1], max_length=26,
                                             padding='max_length', truncation=True)["input_ids"]
            sentence_1_embedding = model(torch.unsqueeze(torch.tensor(tokenized_sentence_1).
                                                         type(torch.LongTensor), 0)).pooler_output.detach().numpy()
            sentence_2_embedding = model(torch.unsqueeze(torch.tensor(tokenized_sentence_2).
                                                         type(torch.LongTensor), 0)).pooler_output.detach().numpy()
        difference = np.abs(sentence_1_embedding - sentence_2_embedding)
        product = sentence_1_embedding * sentence_2_embedding
        pair_embedding = np.concatenate([sentence_1_embedding, sentence_2_embedding, difference, product], axis=0)
        X.append(pair_embedding)
        y.append(one_hot(row["gold_label"]))
    return np.array(X), np.array(y)


def one_hot(label):
    if label == "neutral":
        return np.array([1, 0, 0])
    elif label == "entailment":
        return np.array([0, 1, 0])
    elif label == "contradiction":
        return np.array([0, 0, 1])
    else:
        return None


def save_embeddings(model, data, data_size, embedding_dim=240, X_file_name="", y_file_name="", path=""):
    np.save(path + X_file_name, np.array(np.zeros((data_size, 4, embedding_dim))))
    np.save(path + y_file_name, np.array(np.zeros((data_size, 3))))
    for i in range(data_size // 100):
        X_test_en, y_test_en = np.load(path + X_file_name + ".npy"), np.load(path + y_file_name + ".npy")
        X_data, y_data = get_X_and_y(model, data[i * 100:(i + 1) * 100])
        X_test_en[i * 100:(i + 1) * 100] = X_data
        y_test_en[i * 100:(i + 1) * 100] = y_data
        np.save(path + X_file_name, X_test_en)
        np.save(path + y_file_name, y_test_en)
        print((i + 1) * 100)


def save_train_embeddings(model, data, data_size, embedding_dim=240, output_file_dim=100000,
                          X_file_name="", y_file_name="", path=""):
    os.mkdir(path + X_file_name)
    os.mkdir(path + y_file_name)
    for i in range(data_size // output_file_dim):
        save_embeddings(model=model, data=data[i * output_file_dim:(i + 1) * output_file_dim], data_size=output_file_dim,
                        embedding_dim=embedding_dim, X_file_name=X_file_name + "\\" + str(i),
                        y_file_name=y_file_name + "\\" + str(i), path=path)
        print(i, i, i)


if __name__ == '__main__':
    en_df_train = pd.read_json(path_or_buf="data\\XNLI-1.0\\multinli_1.0_train.jsonl", lines=True)
    test_data = pd.read_csv("data\\XNLI-1.0\\xnli.dev.tsv", sep='\t')
    en_df_test = test_data[test_data["language"] == "en"]
    es_df_test = test_data[test_data["language"] == "es"]
    print("data ready")

    device = torch.device('cuda')
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to(device)
    print("model ready")

    if not os.path.exists("data\\XNLI embeddings"):
        os.mkdir("data\\XNLI embeddings")
    save_train_embeddings(model, en_df_train, 390000, 768, 10000, "X_train_en", "y_train_en", "data\\XNLI embeddings\\")
    save_embeddings(model, en_df_test, 2400, 768, "X_test_en", "y_test_en", "data\\XNLI embeddings\\")
    print("embedding done")
