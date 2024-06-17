import os
import pandas as pd
import numpy as np
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import random
import torch

Bearing_label = ["7_N_3","7_SB_1","7_SI_1","7_SO_1","7_WB_1","7_WI_1","7_WO_1"]
# Bearing_label = ["8_N_2","8_SB_1","8_SI_1","8_SO_1"]
raw_path = "C:/Users/user/Desktop/연구실/데이터/data/2022/6207/"
# raw_path = "C:/Users/user/Desktop/연구실/데이터/data/2022/6208/"

def CSV_READER(Bearing_label, raw_path, desired_file_count):
    Total_DATA = []

    for i in Bearing_label:
        print('현재_' + str(i) + '_진행증')
        directory = os.path.join(raw_path, i)
        mid_data = []

        files_to_read = os.listdir(directory)[:desired_file_count]

        for filename in files_to_read:
            if filename.endswith(".csv"):
                file_path = os.path.join(directory, filename)
                data = pd.read_csv(file_path)
                num_chunks = len(data) // 10240
                for j in range(num_chunks):
                    # [1:2] vibration ,  [2:3] Acoustic
                    chunk = data.iloc[j*10240 : (j+1)*10240, 1:2]
                    mid_data.append(chunk)

        Total_DATA.append(mid_data)

    return Total_DATA

desired_file_count = 3
DATA = CSV_READER(Bearing_label, raw_path, desired_file_count)

device = "cuda" if torch.cuda.is_available() else "cpu"
device

all_data = [item for sublist in DATA for item in sublist]
all_data_np = np.array(all_data)
labels = np.concatenate([np.full(len(class_data), i) for i, class_data in enumerate(DATA)])
X_train, X_test, y_train, y_test = train_test_split(all_data_np, labels, test_size=0.2, random_state=42)
train_dataset = TensorDataset(torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).to(device, dtype=torch.long))
test_dataset = TensorDataset(torch.from_numpy(X_test).to(device), torch.from_numpy(y_test).to(device, dtype=torch.long))

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Model(nn.Module):
    def __init__(self, number_classes=7):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 4, 3, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(4, 8, 3, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(8, 12, 3, padding= 1),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(12*(10240//(4**3)), 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, number_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = Model().to(device)
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_train_loss = total_train_loss / len(train_dataloader)
    train_accuracy = 100.0 * correct_train / total_train

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%")

    if avg_train_loss < best_loss:
        best_loss = avg_train_loss
        best_model_state_dict = model.state_dict()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping....")
            break