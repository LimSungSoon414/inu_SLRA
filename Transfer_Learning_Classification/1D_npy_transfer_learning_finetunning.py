import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import random

# source_data_folder = "C:\\Users\\gyuhee\\Desktop\\data\\CASE_WESTERN_RESERVE\\transfer_DE\\0.007_numpy"
source_data_folder = "C:/Users/user/Desktop/연구실/NPY/similarity"

classes = ['N', 'SB', 'SI', 'SO']

# file_path = "C:/Users/user/Desktop/연구실/similarity/N_numpy/N_0.npy"
#
# data = np.load(file_path)
#
# print(data[:])

def load_data(data_folder):
    data = []
    labels = []

    for i, class_name in enumerate(classes):
        class_folder_path = os.path.join(data_folder, f"{class_name}_numpy")
        npy_files = os.listdir(class_folder_path)

        for npy_file_name in npy_files:
            npy_file_path = os.path.join(class_folder_path, npy_file_name)

            data_chunk = np.load(npy_file_path)
            data.append(data_chunk)
            labels.append(i)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    return data, labels


class Model(nn.Module):
    def __init__(self, number_classes=4):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 4, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(8, 12, 3, padding=1),
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

# class Model(nn.Module):
#     def __init__(self, number_classes=4):
#         super(Model, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv1d(1, 4, 3, padding=1),
#             nn.LeakyReLU(),
#             nn.MaxPool1d(2, 2),
#             nn.Conv1d(4, 8, 3, padding=1),
#             nn.LeakyReLU(),
#             nn.MaxPool1d(2, 2),
#             nn.Conv1d(8, 12, 3, padding=1),
#             nn.LeakyReLU(),
#             nn.MaxPool1d(2, 2),
#         )
#         self.classifier = nn.Sequential(
#             # nn.Dropout(0.3),
#             nn.Linear(12 * (400 // (2 ** 3)), 256),
#             nn.LeakyReLU(),
#             # # nn.Dropout(0.3),
#             nn.Linear(256, 32),
#             nn.LeakyReLU(),
#             # # nn.Dropout(0.3),
#             # nn.Linear(256, 32),
#             # nn.ReLU(),
#             nn.Linear(32, number_classes),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x
def preprocess_data(data):
    return data.reshape(-1, 1, 10240)

X_source, y_source = load_data(source_data_folder)
X_train, X_test, y_train, y_test = train_test_split(X_source, y_source, test_size=0.2, random_state=42)

X_train_all, y_train_all = [], []
num_samples_per_class = 4

for class_id in range(len(classes)):
    class_indices = np.where(y_source == class_id)[0]
    selected_indices = random.sample(class_indices.tolist(), num_samples_per_class)
    X_train_all.extend(X_source[selected_indices])
    y_train_all.extend(y_source[selected_indices])

X_train_all = np.array(X_train_all, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)  # 추가된 부분
X_train_all = preprocess_data(X_train_all)
X_test = preprocess_data(X_test)

y_train_all = np.array(y_train_all, dtype=np.int64)

source_dataset = TensorDataset(torch.from_numpy(X_train_all).to("cuda"), torch.from_numpy(y_train_all).to("cuda"))
source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)

model = Model().to("cuda")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

num_epochs = 100

model.eval()
correct_test = 0
total_test = 0

test_dataset = TensorDataset(torch.from_numpy(X_test).to("cuda"), torch.from_numpy(y_test).to("cuda"))
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in source_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_loss = total_loss / len(source_dataloader)
    train_accuracy = 100.0 * correct_train / total_train

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

torch.save(model.state_dict(), '../1D_6207_model.pth')
