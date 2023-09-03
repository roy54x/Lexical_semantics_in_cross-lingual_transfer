import os
import random
import time
from statistics import mean

import numpy as np
import torch
from torch import nn


class ExMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ExMLP, self).__init__()
        self.ReLU = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.25)
        self.Flatten = torch.nn.Flatten()
        #self.layer1 = MatMul(input_size, hidden_size)
        #self.layer2 = torch.nn.Linear(hidden_size * 4, hidden_size)
        self.layer2 = torch.nn.Linear(input_size * 4, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, output_size)

    def name(self):
        return "MLP"

    def forward(self, x):
        #x = self.layer1(x)
        #x = self.ReLU(x)
        #x = self.dropout(x)
        x = self.Flatten(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.layer3(x)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


class MatMul(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MatMul, self).__init__()
        self.matrix = torch.nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(in_channels, out_channels)),
                                         requires_grad=True)

    def forward(self, x):
        x = torch.matmul(x, self.matrix)
        return x


def get_accuracy(outputs, labels):
    return torch.sum(torch.argmax(outputs, dim=1) == torch.argmax(labels, dim=1)).item() / len(labels)


if __name__ == '__main__':
    device = torch.device("cuda")
    path = "data\\XNLI embeddings\\"
    X_train_paths = [path + "X_train_en\\" + file_name for file_name in
                  os.listdir(path + "X_train_en")]
    y_train_paths = [path + "y_train_en\\" + file_name for file_name in
                  os.listdir(path + "y_train_en")]
    X_test_en = torch.Tensor(np.load(path + "X_test_en.npy")).to(device)
    y_test_en = torch.Tensor(np.load(path + "y_test_en.npy")).to(device)
    print("data ready")

    mlp = ExMLP(768, 3, 128)
    mlp.to(device)

    epochs = 150
    batch_size = 16
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    max_accuracy = 0

    for epoch in range(epochs):
        start_time = time.time()
        train_accuracies = []
        outer_loop = list(range(len(X_train_paths)))
        random.shuffle(outer_loop)
        for i in outer_loop:
            X_data = torch.Tensor(np.load(X_train_paths[i])).to(device)
            y_data = torch.Tensor(np.load(y_train_paths[i])).to(device)
            inner_loop = list(range(len(X_data)//batch_size))
            random.shuffle(inner_loop)
            for j in inner_loop:
                inputs = X_data[j * batch_size:(j + 1) * batch_size]
                labels = y_data[j * batch_size:(j + 1) * batch_size]
                outputs = mlp(inputs)
                loss = criterion(outputs, labels)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_accuracy = get_accuracy(outputs, labels)
                train_accuracies.append(train_accuracy)
        print("epoch " + str(epoch) + " train accuracy is: " + str(mean(train_accuracies)))
        test_accuracy = get_accuracy(mlp(X_test_en), y_test_en)
        print("epoch " + str(epoch) + " test accuracy is: " + str(test_accuracy))
        if test_accuracy > max_accuracy:
            print("the test accuracy is the highest so far, saving model")
            torch.save(mlp, "xnli_model_en_2_128_hidden_layers")
            max_accuracy = test_accuracy
        print("time taken: " + str(time.time()-start_time) + " seconds")
