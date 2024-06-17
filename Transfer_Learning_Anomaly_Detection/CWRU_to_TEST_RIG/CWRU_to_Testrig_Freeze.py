import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import glob
import os
import random
class NPZLoader(data.Dataset): #넘파이 파일을 불러오는 기능
    """Preprocessing을 포함한 dataloader를 구성"""
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        loaded = np.load(self.data[index])
        x = torch.from_numpy(loaded['x'])
        y_data = loaded['y']
        x = x.unsqueeze(0)  # 차원 추가
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

    def __len__(self):
        return len(self.data)

def get_data(args, data_dir=r"C:\Users\user\Desktop\연구실\spectrogram\스펙트로그램_class7\6207\2024\300"):
    """데이터 로더를 생성하고, 데이터를 불러옵니다."""
    classes = ['N', 'SB', 'SI', 'SO', 'WB', 'WI', 'WO']
    normal_class = 'N'

    data_train = []
    data_test = []
    labels_test = []

    for class_name in classes:
        #class_folder = os.path.join(data_dir, class_name+'_NPZ')
        class_folder = os.path.join(data_dir, class_name)
        npz_list = glob.glob(os.path.join(class_folder, '*.npz'))

        if class_name == normal_class:
            npz_list_train = random.sample(npz_list, 5)
            npz_list_test = random.sample([f for f in npz_list if f not in npz_list_train], 100)
            # print(f"loaded {len(npz_list_train)} files from {class_folder} for training")
            # print(f"loaded {len(npz_list_test)} files from {class_folder} for testing")
            data_train.extend(npz_list_train)
            data_test.extend(npz_list_test)
            labels_test.extend([0] * len(npz_list_test))
        else:
            npz_list_test = random.sample(npz_list, 100)
            # print(f"loaded {len(npz_list_test)} files from {class_folder} for testing")
            data_test.extend(npz_list_test)
            labels_test.extend([1] * len(npz_list_test))

    dataset_train = NPZLoader(data_train, [0]*len(data_train))
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"train data load: {len(dataset_train)}")

    dataset_test = NPZLoader(data_test, labels_test)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"test data load: {len(dataset_test)}")

    return dataloader_train, dataloader_test

class DeepSVDD_network(nn.Module):
    def __init__(self, z_dim=32):
        super(DeepSVDD_network, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 16 * 16, z_dim, bias=False)  # Adjust the dimensions according to your input data

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

################
class TrainerDeepSVDD:
    def __init__(self, args, data_loader, device):
        self.args = args
        self.train_loader = data_loader
        self.device = device
        self.net = DeepSVDD_network(z_dim=self.args.z_dim).to(device)
        self.c = None

    def train(self):
        """Deep SVDD model 학습"""

        saved_params = torch.load('C:\\Users\\user\\Desktop\\python\\Transfer_Learning_Anomaly_Detection\\retrained_parameters_Testrig900.pth')
        self.net.load_state_dict(saved_params['net_dict'])
        self.c = torch.Tensor(saved_params['center']).to(self.device)

        for name, param in self.net.named_parameters():
            if name == 'fc1.weight' or name == 'fc1.bias':
                param.requires_grad = True
            else:
                param.requires_grad = False

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.lr_milestones, gamma=0.1)

        self.net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = self.net(x)
                loss = torch.mean(torch.sum((z - self.c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(epoch, total_loss / len(self.train_loader)))

        return self.net, self.c