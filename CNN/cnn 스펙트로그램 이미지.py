import torch
import torchvision.transforms as tr
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder

data_dir = 'C:/Users/user/Desktop/연구실/스펙트로그램_시각화/6207'

transform = tr.Compose([
    tr.Resize((128, 128)),
    tr.ToTensor(),
    tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(root=data_dir, transform=transform)

train_size = int(0.7 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class Model(nn.Module):
    def __init__(self, number_classes=7):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
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
            nn.Dropout(0.3),
            nn.Linear(64 * 16 * 16, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, number_classes),
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

def train(epoch, model, optimizer, loss_func):
    model.train()
    running_loss = 0.0

    for batch, target in train_loader:
        batch, target = batch.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(batch)
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

for epoch in range(num_epochs):
    train_loss = train(epoch, model, optimizer, loss_func)
    test_acc = test(model, test_loader)
    print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

print('Training completed')