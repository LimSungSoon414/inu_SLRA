# 제대로 된 것
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data_folder = r"C:\Users\user\Desktop\NPY"
# data_folder = "C:\\Users\\gyuhee\\Desktop\\data\\bearing\\6028_numpy_0.1"


# classes = ['N', 'SB', 'SI', 'SO', '8_N_2']
classes = ['8_N','8_SB_1','8_SI_1','8_SO_1','8_WB','8_WI','8_WO']
#'8_WB','8_WI','8_WO']
#
sampling_numbers = [10,30,50,70]

results = []

class Model(nn.Module):
    def __init__(self, number_classes=4):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 4, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(4, 8, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(8, 12, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(12 * (10240// (2 ** 3)), 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, number_classes),
            nn.Softmax(dim=1)
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
#             nn.Linear(12 * (12000 // (2 ** 3)), 256),
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

for num_samples in sampling_numbers:
    data = []
    labels = []


    for i, class_name in enumerate(classes):
        class_folder_path = os.path.join(data_folder, f"{class_name}_numpy")
        # class_folder_path = os.path.join(data_folder, f"{class_name}")
        npy_files = os.listdir(class_folder_path)

        for npy_file_name in npy_files[:num_samples]:
            npy_file_path = os.path.join(class_folder_path, npy_file_name)

            data_chunk = np.load(npy_file_path)
            data.append(data_chunk)
            labels.append(i)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    X_train_all, X_test, y_train_all, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)

    X_train_all = X_train_all.reshape(-1, 1, 10240)
    X_test = X_test.reshape(-1, 1, 10240)

    train_dataset = TensorDataset(torch.from_numpy(X_train_all).to("cuda"), torch.from_numpy(y_train_all).to("cuda"))
    test_dataset = TensorDataset(torch.from_numpy(X_test).to("cuda"), torch.from_numpy(y_test).to("cuda"))

    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = Model().to("cuda")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    num_epochs = 25

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_dataloader)
        train_accuracy = 100.0 * correct_train / total_train
        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    model.eval()
    test_loss = 0.0
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    test_loss /= len(test_dataloader)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")

    results.append((num_samples, accuracy))

for num_samples, accuracy in results:
   print(f"Number of Samples: {num_samples}, Test Accuracy: {accuracy:.4f}")
     # print(f"{accuracy:.4f}")
