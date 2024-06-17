import torch
import torchvision.transforms as tr
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import random
from torch.utils.data import random_split
# import warnings
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# warnings.filterwarnings("ignore")

data_dir = 'C:/Users/user/Desktop/연구실/스팩트로그램_1/6207'

max_files_per_folder = 40

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

device = "cuda" if torch.cuda.is_available() else "cpu"
device

random.shuffle(DATA)
train_size = int(0.9 * len(DATA))
test_size = len(DATA) - train_size
train_data = DATA[:train_size]
test_data = DATA[train_size:]
validation_size = int(0.1 * len(train_data))
train_size = len(train_data) - validation_size
train_data_1, validation_data_1 = random_split(train_data, [train_size, validation_size])


# transform = tr.Compose([tr.ToTensor()])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])

class TensorDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx][0]
        y = self.data[idx][1]
        if self.transform:
            x = self.transform(x)
        y = torch.tensor(y,dtype=torch.long)
        return x, y

batch_size = 32
train_dataset = TensorDataset(train_data_1, transform=transform)
test_dataset = TensorDataset(test_data, transform=transform)
validation_dataset = TensorDataset(validation_data_1, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

class Model(nn.Module):
    def __init__(self, number_classes=7):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, number_classes),
        )

    def forward(self, x):
        x = x.to(device).float()
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
model = Model().to(device)


learning_rate = 0.001
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(epoch,train_loader, model, optimizer, loss_func):
    model.train()
    running_loss = 0.0

    for data_tuple in train_loader:
        data = data_tuple[0].to(device)
        target = data_tuple[1].to(device)

        optimizer.zero_grad()

        output = model(data)
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
        for data_tuple in test_loader:
            data = data_tuple[0].to(device)
            target = data_tuple[1].to(device)

            output = model(data)
            values, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()

        accuracy = 100.0 * correct / len(test_loader.dataset)
        return accuracy

num_epochs = 1000
best_acc = 0.0
best_validation_acc = 0.0
patience = 10
min_delta = 0.01
train_losses = []
validation_losses = []
test_accuracies = []
validation_accuracies = []

for epoch in range(num_epochs):
    train_loss = train(epoch,train_loader, model, optimizer, loss_func)
    test_acc = test(model, test_loader)
    validation_acc = test(model, validation_loader)
    validation_loss = test(model, validation_loader)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%,Validation Accuracy: {validation_acc:.2f}%")

    train_losses.append(train_loss)
    test_accuracies.append(test_acc)
    validation_accuracies.append(validation_acc)
    validation_losses.append(validation_loss)
    if test_acc > best_acc:
        best_acc = test_acc
        if validation_loss <= min(validation_losses) - min_delta:
            patience -= 1
            if patience == 0:
                print("Early stopping!")
                break
        else:
            best_validation_acc = validation_acc
print(f'Best Test Accuracy: {best_acc:.2f}%')
print(f'Best Validation Accuracy: {best_validation_acc:.2f}%')
print('Training completed')

# first_sample = next(iter(train_loader))[0]
# channels = first_sample.shape[1]
# print("Number of channels:", channels)
# batch = next(iter(train_loader))
# input_data, labels = batch
# print("Input data shape:", input_data.shape)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), validation_losses, label = 'Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.plot(range(1, num_epochs + 1), validation_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')
plt.legend()

plt.tight_layout()
plt.show()