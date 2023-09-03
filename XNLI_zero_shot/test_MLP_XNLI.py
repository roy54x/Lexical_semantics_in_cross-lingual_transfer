import os
import json
import pandas as pd
import simalign
import torch
from sentence_transformers import SentenceTransformer
from transformers import RobertaModel, RobertaTokenizer, AutoTokenizer
from trankit import Pipeline

from transformations.alphabet_transformation import transform_sentences_alphabet
from transformations.semantic_transformation import transform_sentences_semantic
from transformations.syntax_transformation import transform_sentences_syntax
from save_ground_truth_XNLI import get_X_and_y
from train_MLP_XNLI import ExMLP, get_accuracy

test_data = pd.read_csv("data\\XNLI-1.0\\xnli.dev.tsv", sep='\t')
en_df_test = test_data[test_data["language"] == "en"]
es_df_test = test_data[test_data["language"] == "es"]
el_df_test = test_data[test_data["language"] == "el"]
zh_df_test = test_data[test_data["language"] == "zh"]
print("data ready")

device = torch.device("cuda")
teacher_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
tokenizer = RobertaTokenizer.from_pretrained(r"tokenizers/tokenizer_en_el_cc100_clean")
#tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
student_model = RobertaModel.from_pretrained(r"models\semantic_research_models\sentence_transformer_TED_en_el-semantic_50000_6_layers_6_attention-heads\lowest_loss")
mlp = torch.load(r"C:\Users\RoyIlani\pythonProject\models\xnli_model_en_2_128_hidden_layers").to(device)
mlp.dropout = torch.nn.Dropout(0.0)
print("models ready")

main_dir = r"C:\Users\RoyIlani\pythonProject\data\alignments\en-el_alignments"
with open(
        os.path.join(main_dir, "en-el_alignments_lemmas_only1to1_clean_inverse_clean_inverse.json"),
        "r") as json_file:
    lemmas_dic = json.load(json_file)
with open(os.path.join(main_dir, "en-el_alignments_only1to1_clean_inverse_clean_inverse.json"),
          "r") as json_file:
    word_dic = json.load(json_file)
aligner = simalign.SentenceAligner(model="xlmr", token_type="bpe", matching_methods="a", device="cuda:0")
pipeline = Pipeline("english", gpu=True)
pipeline.add("greek")

X_data, y_data = get_X_and_y(student_model, en_df_test, teacher=False, tokenizer=tokenizer, transform_func=transform_sentences_semantic, target_language="greek",
                             target_df=el_df_test, source_language="english", word_dic=word_dic, lemmas_dic=lemmas_dic, aligner=aligner,
                             pipeline=pipeline)
X_data, y_data = torch.Tensor(X_data).to(device), torch.Tensor(y_data).to(device)
outputs = mlp(X_data)
accuracy = get_accuracy(outputs, y_data)
print("test accuracy is: " + str(round(accuracy, 3)))
