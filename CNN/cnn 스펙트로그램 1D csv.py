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

raw_path = "C:/Users/user/Desktop/연구실/데이터/data/2022/6207/"


def CSV_READER(Bearing_label, raw_path, max_files_per_folder=5):
    Total_DATA = []

    for i in Bearing_label:
        print('현재_' + str(i) + '_진행중')
        directory = os.path.join(raw_path, i)
        mid_data = []

        csv_files = [filename for filename in os.listdir(directory) if filename.endswith(".csv")]

        for filename in csv_files[:max_files_per_folder]:
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)
            # [1:2] vibration ,  [2:3] Acoustic
            arr = data.iloc[:, 1:2].values

            if len(arr) == int(50 * 10240):
                x = arr.reshape(50, 10240)
                mid_data.append(torch.tensor(x, dtype=torch.float32))

        if mid_data:
            Total_DATA.append(mid_data)

    return Total_DATA

DATA = CSV_READER(Bearing_label, raw_path, max_files_per_folder=5)

train_data = []
train_labels = []
test_data = []
test_labels = []

for i, label_data in enumerate(DATA):
    train_segment, test_segment = train_test_split(label_data, test_size=0.3, random_state=42)


    for j in train_segment:
        train_data.append(j)
        train_labels.append(i)


    for k in test_segment:
        test_data.append(k)
        test_labels.append(i)


train_data_tensor = torch.stack(train_data)
train_labels_tensor = torch.tensor(train_labels)
test_data_tensor = torch.stack(test_data)
test_labels_tensor = torch.tensor(test_labels)

batch_size = 32
train_dataset = TensorDataset(train_data_tensor, train_labels_tensor)
test_dataset = TensorDataset(test_data_tensor, test_labels_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Model(nn.Module):
    def __init__(self, number_classes=7):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            # nn.Conv1d(1, 50, 5, padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(2, 2),
            nn.Conv1d(50, 70, 5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(70, 85, 5, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(85*(50//(2**2))*(10240//(2**2)), 4096),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, number_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model().to(device)
learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# print(test_data_tensor[0].shape)
# print(test_labels_tensor.shape)
# print(train_data_tensor.shape)
# print(train_labels_tensor.shape)
# first_sample = next(iter(train_loader))[0]
# channels = first_sample.shape[1]
# print("Number of channels:", channels)
# batch = next(iter(train_loader))
# input_data, labels = batch
# print("Input data shape:", input_data.shape)


def train(epoch, model, optimizer, loss_func,train_loader):
    model.train()
    running_loss = 0.0

    for batch, target in train_loader:
        batch, target = batch.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(batch)
        target = target.long()
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss

def test(model, test_loader):
    model.eval()
    correct = 0

    with torch.no_grad():
        for batch, target in test_loader:
            batch, target = batch.to(device), target.to(device)

            output = model(batch)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        return accuracy

num_epochs = 10
best_acc = 0.0
for epoch in range(num_epochs):
    train_loss = train(epoch, model, optimizer, loss_func, train_loader)
    test_acc = test(model, test_loader)
    with torch.no_grad():
        test_acc = test(model, test_loader)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    if test_acc > best_acc:
        best_acc = test_acc

print(f'Best Test Accuracy: {best_acc:.2f}%')
print('Training completed')
