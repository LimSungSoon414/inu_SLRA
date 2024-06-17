import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import easydict
import torch
import torch.nn as nn
import torch.nn.functional as F

train_file = 'C:\\Users\\user\\Desktop\\open\\train.csv'
test_file =  'C:\\Users\\user\\Desktop\\open\\test.csv'
train_folder = 'C:\\Users\\user\\Desktop\\open\\train'
test_folder = 'C:\\Users\\user\\Desktop\\open\\test'

class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, train=True, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.dataframe.iloc[idx, 0] + '.png')

        if self.train:
            label = self.dataframe.iloc[idx, 2]
        else:
            label = None

        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if label is not None:
            return image, label
        else:
            return image, 0  # 이상치 예측값으로 0 반환 혹은 다른 값을 지정

def global_contrast_normalization(image):
    """Apply global contrast normalization to tensor. """
    mean = torch.mean(image)  # mean over all features (pixels) per sample
    image -= mean
    image_scale = torch.mean(torch.abs(image))
    image /= image_scale
    return image

def get_data(train_file,test_file,train_folder,test_folder,args):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: global_contrast_normalization(x)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_data = pd.read_csv(train_file)
    train_dataset = CustomDataset(dataframe=train_data, root_dir=train_folder, train=True, transform=transform)

    test_data = pd.read_csv(test_file)
    test_dataset = CustomDataset(dataframe=test_data, root_dir=test_folder, train=False, transform=transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader

class DeepSVDD_network(nn.Module):
    def __init__(self, z_dim=32):
        super(DeepSVDD_network, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 12, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(12, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(12, 8, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 16 * 16, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

class pretrain_autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(pretrain_autoencoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(3, 12, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(12, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(12, 8, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv3 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 16 * 16, z_dim, bias=False)

        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=2)
        self.bn5 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 12, 5, bias=False, padding=2)
        self.bn6 = nn.BatchNorm2d(12, eps=1e-04, affine=False)
        self.deconv4 = nn.ConvTranspose2d(12, 3, 5, bias=False, padding=2)

    def encoder(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = self.conv3(x)
        x = self.pool(F.leaky_relu(self.bn3(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decoder(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=4)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn5(x)), scale_factor=2)
        x = self.deconv3(x)
        x = F.interpolate(F.leaky_relu(self.bn6(x)), scale_factor=2)
        x = self.deconv4(x)
        return torch.sigmoid(x)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class TrainerDeepSVDD:
    def __init__(self, args, data_loader, device):
        self.args = args
        self.train_loader = data_loader
        self.device = device

    def pretrain(self):
        ae = pretrain_autoencoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = torch.optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                                     weight_decay=self.args.weight_decay_ae)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         milestones=self.args.lr_milestones, gamma=0.1)

        ae.train()
        for epoch in range(1,self.args.num_epochs_ae+1):
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
                    'net_dict': net.state_dict()}, 'C:/Users/user/Desktop/python/Deep SVDD/pretrained_parameters_Dacon.pth')

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
            state_dict = torch.load('C:/Users/user/Desktop/python/Deep SVDD/pretrained_parameters_Dacon.pth')
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
        for epoch in range(1,self.args.num_epochs+1):
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
