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

    data_dir = 'C:/Users/user/Desktop/연구실/spectrogram/스펙트로그램_class7/6207/2022'

    max_files_per_folder = 100

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

    class_data = {}
    for x, y in DATA:

        class_label = str(y)
        if class_label not in class_data:
            class_data[class_label] = []
        class_data[class_label].append((x, y))


    train_data = []
    test_data = []
    for class_label, data_list in class_data.items():
        random.shuffle(data_list)
        split_index = int(0.6* len(data_list))
        train_data.extend(data_list[:split_index])
        test_data.extend(data_list[split_index:])


    # random.shuffle(DATA)
    # train_size = int(0.7 * len(DATA))
    # test_size = len(DATA) - train_size
    # train_data = DATA[:train_size]
    # test_data = DATA[train_size:]

    # transform = tr.Compose([tr.ToTensor()])
    transform = transforms.Compose([
        transforms.ToTensor(),
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
    train_dataset = TensorDataset(train_data, transform=transform)
    test_dataset = TensorDataset(test_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    class Model(nn.Module):
        def __init__(self, number_classes=7):
            super(Model, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 4, 3, padding=1),
                # nn.LeakyReLU(),
                nn.LeakyReLU(),
                # nn.Tanh(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(4, 8, 3, padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(8, 16, 3, padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.LeakyReLU(),
                nn.MaxPool2d(2, 2),
            )
            self.classifier = nn.Sequential(
                # nn.Dropout(0.3),
                nn.Linear(32 * 4 * 4, 256),
                nn.LeakyReLU(),
                # # nn.Dropout(0.3),
                nn.Linear(256, 32),
                nn.LeakyReLU(),
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


    learning_rate = 0.00005
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

    num_epochs = 50
    best_acc = 0.0

    train_losses = []
    test_accuracies = []

    for epoch in range(num_epochs):
        train_loss = train(epoch,train_loader, model, optimizer, loss_func)
        test_acc = test(model, test_loader)
        print(f"Epoch: {epoch+1}, Train Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

        train_losses.append(train_loss)
        test_accuracies.append(test_acc)

        if test_acc > best_acc:
            best_acc = test_acc

    print(f'Best Test Accuracy: {best_acc:.2f}%')
    print('Training completed')

    #
    # torch.save(model, 'entire_model.pth')
    #
    # npz_file_path = 'C:/Users/user/Desktop/연구실/스펙트로그램_2/6208'
    # max_files_per_folder_1 = 10
    # def npz_reader_1(npz_file_path):
    #
    #     loaded_data = []
    #
    #
    #     for class_folder in os.listdir(npz_file_path):
    #         class_folder_path = os.path.join(npz_file_path, class_folder)
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
    # DATA_1 = npz_reader_1(npz_file_path)
    #
    # class_data_1 = {}
    # for x, y in DATA_1:
    #
    #     class_label = str(y)
    #     if class_label not in class_data_1:
    #         class_data_1[class_label] = []
    #     class_data_1[class_label].append((x, y))
    #
    # train_data_1 = []
    # test_data_1 = []
    # for class_label, j in class_data_1.items():
    #     random.shuffle(j)
    #     split_index = int(0.8 * len(j))
    #     train_data_1.extend(j[:split_index])
    #     test_data_1.extend(j[split_index:])
    #
    # batch_size = 32
    # train_dataset_1 = TensorDataset(train_data_1, transform=transform)
    # test_dataset_1 = TensorDataset(test_data_1, transform=transform)
    # train_loader_1 = DataLoader(train_dataset_1, batch_size=batch_size, shuffle=True)
    # test_loader_1 = DataLoader(test_dataset_1, batch_size=batch_size, shuffle=False)
    #
    # loaded_model = torch.load('entire_model.pth')
    # loaded_model.eval()
    #
    # # for param in loaded_model.parameters():
    # #     print(param)
    # def train(epoch,train_loader_1, loaded_model, optimizer, loss_func):
    #     loaded_model.train()
    #     running_loss = 0.0
    #
    #     for data_tuple in train_loader_1:
    #         data = data_tuple[0].to(device)
    #         target = data_tuple[1].to(device)
    #
    #         optimizer.zero_grad()
    #
    #         output = loaded_model(data)
    #         loss = loss_func(output, target)
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_loss += loss.item()
    #
    #     epoch_loss = running_loss / len(train_loader_1)
    #     return epoch_loss
    #
    # def test(loaded_model, test_loader_1):
    #     loaded_model.eval()
    #     correct_1 = 0
    #
    #     with torch.no_grad():
    #         for data_tuple_1 in test_loader_1:
    #             data = data_tuple_1[0].to(device)
    #             target = data_tuple_1[1].to(device)
    #             output = loaded_model(data)
    #             values, predicted = torch.max(output.data, 1)
    #             correct_1 += (predicted == target).sum().item()
    #
    #         accuracy_1 = 100.0 * correct_1 / len(test_loader_1.dataset)
    #         return accuracy_1
    #
    # train_losses_1 = []
    # test_accuracies_1 = []
    # best_acc_1 = 0.0
    # for epoch_1 in range(num_epochs):
    #     train_loss_1 = train(epoch_1,train_loader_1, loaded_model, optimizer, loss_func)
    #     test_acc_1 = test(loaded_model, test_loader_1)
    #     print(f"Epoch: {epoch_1+1}, Train Loss: {train_loss_1:.4f}, Test Accuracy: {test_acc_1:.2f}%")
    #
    #     train_losses_1.append(train_loss_1)
    #     test_accuracies_1.append(test_acc_1)
    #
    #     if test_acc_1 > best_acc_1:
    #         best_acc_1 = test_acc_1
    #
    # print(f'Best Test Accuracy: {best_acc_1:.2f}%')
    # print('Training completed')
    #
    # #
    # # first_sample = next(iter(train_loader))[0]
    # # channels = first_sample.shape[1]
    # # print("Number of channels:", channels)
    # # batch = next(iter(train_loader))
    # # input_data, labels = batch
    # # print("Input data shape:", input_data.shape)
    # #
    # # plt.figure(figsize=(10, 5))
    # # plt.subplot(1, 2, 1)
    # # plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Loss')
    # # plt.title('Training Loss')
    # # plt.legend()
    # #
    # # plt.subplot(1, 2, 2)
    # # plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    # # plt.xlabel('Epoch')
    # # plt.ylabel('Accuracy (%)')
    # # plt.title('Test Accuracy')
    # # plt.legend()
    # #
    # # plt.tight_layout()
    # # plt.show()
    # # from sklearn.metrics import confusion_matrix
    # # import seaborn as sns
    # #
    # #
    # # loaded_model.eval()
    # # true_labels = []
    # # predicted_labels = []
    # #
    # # with torch.no_grad():
    # #     for data_tuple in test_loader_1:
    # #         data = data_tuple[0].to(device)
    # #         target = data_tuple[1].to(device)
    # #
    # #         output = loaded_model(data)
    # #         _, predicted = torch.max(output, 1)
    # #
    # #         true_labels.extend(target.cpu().numpy())
    # #         predicted_labels.extend(predicted.cpu().numpy())
    # #
    # #
    # # conf_matrix = confusion_matrix(true_labels, predicted_labels)
    # #
    # # # class_names = ["N", "SB", "SI", "SO", "WB", "WI", "WO"]
    # # class_names = ["N", "SB", "SI", "SO"]
    # #
    # # plt.figure(figsize=(8, 6))
    # # sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    # # plt.xlabel("Predicted")
    # # plt.ylabel("True")
    # # plt.title("Confusion Matrix")
    # # plt.show()