import numpy as np
import torch
import easydict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import glob
import os

class NPZLoader(data.Dataset): #넘파이 파일을 불러오는 기능
    """Preprocessing을 포함한 dataloader를 구성"""
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index): #각 인덱스에 해당하는 데이터를 불러옴
        x = torch.from_numpy(np.load(self.data[index])['spectrogram'])
        x = x.unsqueeze(0)
        #print(f"loaded data from {self.data[index]}")
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

import random

def get_data(train_n, args, data_dir = "C:\\Users\\gyuhee\\Desktop\\연구실\\dfdsf"):
    classes = ['N', 'SB', 'SI', 'SO', 'WB', 'WI', 'WO']
    normal_class = 'N'

    data_train = []
    data_test = []
    labels_test = []

    for class_name in classes:
        class_folder = os.path.join(data_dir, class_name+'_NPZ')
        npz_list = glob.glob(os.path.join(class_folder, '*.npz'))
        random.shuffle(npz_list)  # 리스트 랜덤하게 섞기

        if class_name == normal_class:
            npz_list_train = npz_list[:train_n]
            data_train.extend(npz_list_train)
            npz_list_test = npz_list[train_n:train_n+train_n]
            data_test.extend(npz_list_test)
            labels_test.extend([0]*len(npz_list_test))
        else:
            npz_list_test = npz_list[:train_n]
            data_test.extend(npz_list_test)
            labels_test.extend([1]*len(npz_list_test))

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


class pretrain_autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(pretrain_autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 16 * 16, z_dim, bias=False)  # Adjust the dimensions according to your input data

        self.deconv1 = nn.ConvTranspose2d(32, 4, 4, bias=False,  stride=2, padding=1)  # 수정된 부분
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 4, bias=False,  stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 4, bias=False, stride=2, padding=1)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # print(x.size())  # print the size of output tensor
        return x

    def decoder(self, x):
        x = x.view(x.size(0), self.z_dim, 1, 1)  # Adjust the dimensions according to your input data
        #print("After view:", x.size())  # 추가된 부분
        x = F.interpolate(F.leaky_relu(x), scale_factor=(2, 2))
        #print("After first interpolate:", x.size())
        x = self.deconv1(x)
        #print("After first deconv:", x.size())
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=(2, 2))
        #print("After second interpolate:", x.size())
        x = self.deconv2(x)
        #print("After second deconv:", x.size())
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=(2, 2))
        #print("After third interpolate:", x.size())
        x = self.deconv3(x)
        #print("After third deconv:", x.size())
        return torch.sigmoid(x)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat



################
class TrainerDeepSVDD:
    def __init__(self, args, data_loader, device):
        self.args = args
        self.train_loader = data_loader
        self.device = device

    def train(self):
        """Deep SVDD model 학습"""
        net = DeepSVDD_network().to(self.device)

        state_dict = torch.load('C:/Users/gyuhee/PycharmProjects/pythonProject/model_save/best_acc_parameters.pth')
        net.load_state_dict(state_dict['net_dict'])
        c = torch.Tensor(state_dict['center']).to(self.device)
        #r = torch.Tensor(state_dict['r']).to(self.device)  # r값 로드
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,
                                     weight_decay=self.args.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.lr_milestones, gamma=0.1)
        net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                z = net(x)
                loss = torch.mean(torch.sum((z - c) ** 2, dim=1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            scheduler.step()
            print('Training Deep SVDD... Epoch: {}, Loss: {:.3f}'.format(
                epoch, total_loss / len(self.train_loader)))
        self.net = net
        self.c = c
        return self.net, self.c




#         #################학습
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# args = easydict.EasyDict({
#        'num_epochs':10,
#        'num_epochs_ae':10,
#        'lr':1e-3,
#        'lr_ae':1e-3,
#        'weight_decay':5e-7,
#        'weight_decay_ae':5e-3,
#        'lr_milestones':[50],
#        'batch_size':32,
#        'pretrain':True,
#        'latent_dim':32,
#        'normal_class':0
#                 })
#
# if __name__ == '__main__':
#
#     # Train/Test Loader 불러오기
#     dataloader_train, dataloader_test = get_data(args)
#
#     # Network 학습준비, 구조 불러오기
#     deep_SVDD = TrainerDeepSVDD(args, dataloader_train, device)
#
#     # DeepSVDD를 위한 DeepLearning pretrain 모델로 Weight 학습
#     if args.pretrain:
#         deep_SVDD.pretrain()
#
#     # 학습된 가중치로 Deep_SVDD모델 Train
#     net, c = deep_SVDD.train()
#
#
#
# # ##################
# def eval(net, c, dataloader, device):
#     """Testing the Deep SVDD model"""
#
#     scores = []
#     labels = []
#     net.eval()
#     print('Testing...')
#     with torch.no_grad():
#         for x, y in dataloader:
#             x = x.float().to(device)
#             z = net(x)
#             score = torch.sum((z - c) ** 2, dim=1)
#
#             scores.append(score.detach().cpu())
#             labels.append(y.cpu())
#     labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
#     print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
#     return labels, scores