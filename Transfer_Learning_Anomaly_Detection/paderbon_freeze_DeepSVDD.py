import numpy as np
import torch
import easydict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score
import glob
import os
import random

class NPZLoader(Dataset):
    """Preprocessing을 포함한 dataloader를 구성"""

    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        loaded = np.load(self.data[index])
        x = torch.from_numpy(loaded['x'])
        y_data = loaded['y']
        x = x.unsqueeze(0)
        y = self.target[index]

        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)

def get_data(args, data_dir=r"C:\Users\user\Desktop\연구실\spectrogram\paderbon_spectrogram\npz"):
    """get dataloaders"""
    # anomaly_dirs = ['PU_AIRD', 'PU_AORD', 'RIRD', 'RORD']
    # anomaly_subdirs = ['KI01', 'KA01', 'KI04', 'KA04']
    anomaly_dirs = ['PU_AIRD', 'PU_AORD', 'RIRD', 'RORD']
    anomaly_subdirs = ['KI03', 'KA07', 'KI18', 'KA30']

    data_train = []
    data_test = []
    labels_test = []

    # 정상 데이터 로딩
    class_folder = os.path.join(data_dir, 'Normal_target_K002')
    npz_list = glob.glob(os.path.join(class_folder, '*.npz'))
    random.shuffle(npz_list)
    npz_list_train = npz_list[:65]  # 훈련용 데이터
    npz_list_test = npz_list[65:130]  # 테스트용 데이터
    data_train.extend(npz_list_train)
    data_test.extend(npz_list_test)
    labels_test.extend([0] * len(npz_list_test))

    # 이상 데이터 로딩
    for anomaly_dir, anomaly_subdir in zip(anomaly_dirs, anomaly_subdirs):
        class_folder = os.path.join(data_dir, anomaly_dir, anomaly_subdir)
        npz_list_test = glob.glob(os.path.join(class_folder, '*.npz'))[:65]  # 테스트용 데이터
        random.shuffle(npz_list_test)
        data_test.extend(npz_list_test)
        labels_test.extend([1] * len(npz_list_test))

    dataset_train = NPZLoader(data_train, [0] * len(data_train))
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    dataset_test = NPZLoader(data_test, labels_test)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"train data load: {len(dataset_train)}")
    print(f"test data load: {len(dataset_test)}")

    return dataloader_train, dataloader_test


# args = easydict.EasyDict({
#     "batch_size": 32
# })
#
# dataloader_train, dataloader_test = get_data(args)
#
# print("Train data files:")
# for i, data in enumerate(dataloader_train.dataset.data):
#     print(f"Index {i}: {data}")
#
# print("\nTest data files:")
# for i, data in enumerate(dataloader_test.dataset.data):
#     print(f"Index {i}: {data}")


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

    def pretrain(self):
        """ DeepSVDD 모델에서 사용할 가중치를 학습시키는 AutoEncoder 학습 단계"""
        ae = pretrain_autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                                     weight_decay=self.args.weight_decay_ae)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.lr_milestones, gamma=0.1)

        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in self.train_loader:
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = ae(x)
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()

                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                epoch, total_loss / len(self.train_loader)))
        self.save_weights_for_DeepSVDD(ae, self.train_loader)

    def save_weights_for_DeepSVDD(self, model, dataloader):
        """학습된 AutoEncoder 가중치를 DeepSVDD모델에 Initialize해주는 함수"""
        c = self.set_c(model, dataloader)
        net = DeepSVDD_network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, '/Transfer_Learning_Anomaly_Detection/K002.pth')



    def set_c(self, model, dataloader, eps=0.1):
        """Initializing the center for the hypersphere"""
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encoder(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def train(self):
        """Deep SVDD model 학습"""
        net = DeepSVDD_network().to(self.device)

        if self.args.pretrain == True:
            state_dict = torch.load('C:/Users/user/Desktop/python/Transfer_Learning_Anomaly_Detection/K002.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)

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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)

