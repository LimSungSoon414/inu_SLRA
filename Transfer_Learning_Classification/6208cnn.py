import torch
import torchvision.transforms as tr
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from torch.utils.data import random_split
import matplotlib
from sklearn.metrics import accuracy_score
matplotlib.use('TkAgg')
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data_dir =  'C:/Users/user/Desktop/연구실/CASE_WESTERN_RESERVE/transfer_DE_spectrogram/0.007'

max_files_per_folder = 50

def npz_reader(data_dir):

    loaded_data = []

    for class_folder in os.listdir(data_dir):
        class_folder_path = os.path.join(data_dir, class_folder)

        npz_files = [f for f in os.listdir(class_folder_path) if f.endswith(".npz")]

        npz_files = npz_files[:max_files_per_folder]

        for npz_file in npz_files:
            npz_path = os.path.join(class_folder_path, npz_file)
            loaded_npz = np.load(npz_path)

            x = loaded_npz['x']
            y = loaded_npz['y']

            loaded_data.append((x, y))
    return loaded_data
DATA = npz_reader(data_dir)

data = np.array([item[0] for item in DATA])
labels = np.array([item[1] for item in DATA])

device = "cuda" if torch.cuda.is_available() else "cpu"
device

X_train_all, X_test, y_train_all, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, stratify=y_train_all, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.from_numpy(X_train_all).to(device), torch.from_numpy(y_train_all).to(device, dtype=torch.long))
test_dataset = TensorDataset(torch.from_numpy(X_test).to(device), torch.from_numpy(y_test).to(device, dtype=torch.long))
# val_dataset = TensorDataset(torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device, dtype=torch.long))

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


class Model(nn.Module):
    def __init__(self, number_classes=4):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),
            # nn.LeakyReLU(),
            nn.ReLU(),
            # nn.Tanh(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(0.3),
            nn.Linear(32 * 4 * 4, 256),
            nn.ReLU(),
            # # nn.Dropout(0.3),
            nn.Linear(256, 32),
            nn.ReLU(),
            # # nn.Dropout(0.3),
            # nn.Linear(256, 32),
            # nn.ReLU(),
            nn.Linear(32, number_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = x.to(device).float()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
model = Model().to(device)


learning_rate = 0.0001
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_model_state_dict = None
best_loss = float('inf')
patience = 100000
counter = 0
best_acc = 0.0
train_losses = []
test_accuracies = []
# 에폭 수 설정
num_epochs = 50
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

    model.eval()
    # total_val_loss = 0.0
    # correct_val = 0
    # total_val = 0
    #
    # with torch.no_grad():
    #     for inputs, labels in val_dataloader:
    #         outputs = model(inputs.unsqueeze(1))
    #         loss = loss_func(outputs, labels)
    #         total_val_loss += loss.item()
    #
    #         _, predicted = torch.max(outputs.data, 1)
    #         total_val += labels.size(0)
    #         correct_val += (predicted == labels).sum().item()
    #
    # avg_val_loss = total_val_loss / len(val_dataloader)
    # val_accuracy = 100.0 * correct_val / total_val

    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    print(f"Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%,")
          # f" Val_Loss: {avg_val_loss:.4f}, Val_Acc: {val_accuracy:.2f}%")

    train_losses.append(avg_train_loss)
    test_accuracies.append(train_accuracy)

    if train_accuracy > best_acc:
        best_acc = train_accuracy
    # if avg_val_loss < best_loss:
    #     best_loss = avg_val_loss
    #     best_model_state_dict = model.state_dict()
    #     counter = 0
print(f'Best Test Accuracy: {best_acc:.2f}%')
print('Training completed')
#
# if avg_val_loss < best_loss:
#         best_loss = avg_val_loss
#         best_model_state_dict = model.state_dict()
#         counter = 0
#     else:
#         counter += 1
#         if counter >= patience:
#             print("Early stopping....")
#             break
# torch.save(best_model_state_dict, "6207_model.pth")
#
#
# npz_file_path_1 = 'C:/Users/user/Desktop/연구실/CASE_WESTERN_RESERVE/transfer_FE_spectrogram'
#
# max_files_per_folder_1 = 110
#
# def npz_reader_1(npz_file_path_1):
#
#     loaded_data = []
#
#
#     for class_folder in os.listdir(npz_file_path_1):
#         class_folder_path = os.path.join(npz_file_path_1, class_folder)
#
#         npz_files = [f for f in os.listdir(class_folder_path) if f.endswith(".npz")]
#
#         npz_files = npz_files[:max_files_per_folder_1]
#
#         for npz_file in npz_files:
#             npz_path = os.path.join(class_folder_path, npz_file)
#             loaded_npz = np.load(npz_path)
#
#             x = loaded_npz['x']
#             y = loaded_npz['y']
#
#             loaded_data.append((x, y))
#     return loaded_data
#
# DATA_1 = npz_reader_1(npz_file_path_1)
#
# model.load_state_dict(torch.load("6207_model.pth"))
# model.eval()
#
# data_1 = np.array([item[0] for item in DATA_1])
# labels_1 = np.array([item[1] for item in DATA_1])
#  X_test_1, y_test_1 = data_1, labels_1
#
# test_dataset_1 = TensorDataset(torch.from_numpy(X_test_1).to(device),
#                                torch.from_numpy(y_test_1).to(device, dtype=torch.long))
# test_dataloader_1 = DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False)
# print(f"6208 Test 데이터 개수: {len(test_dataset_1)}")
#
# # 6208 모델 로드 및 테스트
# model.load_state_dict(torch.load("6207_model.pth"))
# model.eval()
#
# test_loss = 0.0
# y_true, y_pred = [], []
# with torch.no_grad():
#     for inputs, labels in test_dataloader_1:
#         outputs = model(inputs.unsqueeze(1))
#         loss = loss_func(outputs, labels)
#         test_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         y_true.extend(labels.cpu().numpy())
#         y_pred.extend(predicted.cpu().numpy())
#
# test_loss /= len(test_dataloader_1)
# accuracy = accuracy_score(y_true, y_pred)
#
# print(f"6208 Test - Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")