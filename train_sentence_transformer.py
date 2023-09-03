import os
import statistics
import time

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoTokenizer

from dataset_class import Dataset
from train_sentence_transformer_cfg import get_cfg

cfg = get_cfg()
device = torch.device('cuda:0')
if not cfg["use_tokenizer"]:
    teacher_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    tokenizer = teacher_model.tokenizer
else:
    tokenizer = RobertaTokenizer.from_pretrained(os.path.join("tokenizers", cfg["tokenizer_name"]))
    #tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

config = RobertaConfig(
    vocab_size=len(tokenizer.get_vocab()),
    max_position_embeddings=28,
    hidden_size=768,
    num_attention_heads=cfg["num_attention_heads"],
    num_hidden_layers=cfg["num_hidden_layers"],
    type_vocab_size=1
)

student_model = RobertaModel(config).to(device)
if cfg["parallel_training"]:
    parallel_net = torch.nn.DataParallel(student_model, device_ids=[0, 1, 2, 3, 4, 5, 6])
optim = torch.optim.AdamW(student_model.parameters(), lr=cfg["learning_rate"])
loss_fn = torch.nn.CosineEmbeddingLoss()

input_ids, embeddings = [], []
for file_name in os.listdir(cfg["train_data_path"]):
    if len(input_ids) >= cfg["train_data_size"]:
        break
    print("loading file named: " + file_name)
    file_df = pd.read_json(os.path.join(cfg["train_data_path"], file_name))
    if not cfg["use_tokenizer"]:
        input_ids.extend(file_df.input_ids.tolist())
    else:
        if cfg["col_to_use"] not in file_df.columns:
            continue
        tokenized_sentences = tokenizer(file_df[cfg["col_to_use"]].tolist(), max_length=26, padding='max_length', truncation=True)
        input_ids.extend(tokenized_sentences["input_ids"])
    embeddings.extend(file_df.embedding.tolist())
input_ids = torch.tensor(input_ids)
embeddings = torch.tensor(embeddings)
encodings = {'input_ids': input_ids, 'embeddings': embeddings}
dataset = Dataset(encodings)
train_set, val_set = torch.utils.data.random_split(dataset, [int(cfg["train_data_size"] * (1 - cfg["validation_size"])),
                                                             int(cfg["train_data_size"] * cfg["validation_size"])])
train_loader = DataLoader(train_set, batch_size=cfg["batch_size"], shuffle=True)
train_loop = tqdm(train_loader, leave=True)
val_loader = DataLoader(val_set, batch_size=cfg["batch_size"], shuffle=True)
val_loop = tqdm(val_loader, leave=True)

if not os.path.exists(cfg["model_name"]):
    os.mkdir(cfg["model_name"])
results_file = open(os.path.join(cfg["model_name"], f'results_file'), "w")
lowest_loss = 1
epochs_from_lowest_loss = 0
epoch_idx = 0
while epochs_from_lowest_loss < cfg["epochs_to_decide"]:

    print("Beginning epoch number: " + str(epoch_idx))
    cur_time = time.time()
    epoch_losses = []
    for batch_index, batch in enumerate(train_loop):
        optim.zero_grad()
        input_ids = batch['input_ids'].type(torch.LongTensor).to(device)
        embeddings = batch['embeddings'].to(device)
        if not cfg["parallel_training"]:
            outputs = torch.nn.functional.normalize(student_model(input_ids).pooler_output)
            loss = loss_fn(outputs, embeddings, torch.ones(outputs.shape[0]).to(device))
            loss.backward()
            optim.step()
            batch_loss = loss.item()
        else:
            outputs = torch.nn.functional.normalize(parallel_net(input_ids).pooler_output)
            loss = loss_fn(outputs, embeddings, torch.ones(outputs.shape[0]).to(device))
            loss.mean().backward()
            optim.step()
            batch_loss = loss.mean().item()
        epoch_losses.append(batch_loss)
    output_line = "Epoch " + str(epoch_idx) + " training loss is: " + \
                  str(round(statistics.mean(epoch_losses), 3))
    time_output_line = "Epoch " + str(epoch_idx) + " finished, time taken: " + str((time.time() - cur_time) / 60) \
                       + " minutes" + "\n"
    results_file.write(output_line)
    results_file.write(time_output_line)
    print(output_line)
    #print(time_output_line)

    val_losses = []
    for batch_index, batch in enumerate(val_loop):
        input_ids = batch['input_ids'].type(torch.LongTensor).to(device)
        embeddings = batch['embeddings'].to(device)
        outputs = torch.nn.functional.normalize(student_model(input_ids).pooler_output)
        loss = loss_fn(outputs, embeddings, torch.ones(outputs.shape[0]).to(device))
        batch_loss = loss.item()
        val_losses.append(batch_loss)
    epoch_val_loss = round(statistics.mean(val_losses), 3)
    output_line = "Epoch " + str(epoch_idx) + " VALIDATION loss is: " + \
                  str(epoch_val_loss) + "\n"
    results_file.write(output_line)
    print(output_line)
    if epoch_val_loss < lowest_loss:
        lowest_loss = epoch_val_loss
        student_model.save_pretrained(os.path.join(cfg["model_name"], "lowest_loss"))
        print("THIS IS THE EPOCH WITH THE LOWEST LOSS \n")
        epochs_from_lowest_loss = 0
    else:
        epochs_from_lowest_loss += 1

    if epoch_idx % 50 == 0:
        student_model.save_pretrained(os.path.join(cfg["model_name"], "model_epoch_" + str(epoch_idx)))
    epoch_idx += 1

student_model.save_pretrained(os.path.join(cfg["model_name"], "last_epoch"))

print("Testing the model")
student_model = RobertaModel.from_pretrained(os.path.join(cfg["model_name"], "lowest_loss")).to(device)
input_ids, embeddings = [], []
for file_name in os.listdir(cfg["test_data_path"]):
    if len(input_ids) >= cfg["test_data_size"]:
        break
    file_df = pd.read_json(os.path.join(cfg["test_data_path"], file_name))
    if not cfg["use_tokenizer"]:
        input_ids.extend(file_df.input_ids.tolist())
    else:
        tokenized_sentences = tokenizer(file_df[cfg["col_to_use"]].tolist(), max_length=26,
                                        padding='max_length', truncation=True)
        input_ids.extend(tokenized_sentences["input_ids"])
    embeddings.extend(file_df.embedding.tolist())
input_ids = torch.tensor(input_ids)
embeddings = torch.tensor(embeddings)
encodings = {'input_ids': input_ids, 'embeddings': embeddings}
test_set = Dataset(encodings)
test_loader = DataLoader(test_set, batch_size=cfg["batch_size"], shuffle=True)
test_loop = tqdm(test_loader, leave=True)
test_losses = []
for batch_index, batch in enumerate(test_loop):
    input_ids = batch['input_ids'].type(torch.LongTensor).to(device)
    embeddings = batch['embeddings'].to(device)
    outputs = torch.nn.functional.normalize(student_model(input_ids).pooler_output)
    loss = loss_fn(outputs, embeddings, torch.ones(outputs.shape[0]).to(device))
    batch_loss = loss.item()
    test_losses.append(batch_loss)
test_loss = round(statistics.mean(test_losses), 3)
output_line = "TEST loss is: " + str(test_loss)
results_file.write(output_line)
print(output_line)
results_file.close()
