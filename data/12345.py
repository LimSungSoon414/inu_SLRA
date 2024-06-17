import numpy as np
import easydict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score


class MNIST_Dataset(data.Dataset):

    # Mnist data 처리후 transform 적용함
    def __init__(self, data, target, transform):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(x.numpy(), mode='L')
            x = self.transform(x)
        return x, y


def mnist_loader(args, data_dir='C:/Users/user/Desktop/연구실/MNIST/MNIST'):
    # mnist dataloader
    # min, max values for each class after applying GCN (as the original implementation)
    min_max = [(-0.8826567065619495, 9.001545489292527),
               (-0.6661464580883915, 20.108062262467364),
               (-0.7820454743183202, 11.665100841080346),
               (-0.7645772083211267, 12.895051191467457),
               (-0.7253923114302238, 12.683235701611533),
               (-0.7698501867861425, 13.103278415430502),
               (-0.778418217980696, 10.457837397569108),
               (-0.7129780970522351, 12.057777597673047),
               (-0.8280402650205075, 10.581538445782988),
               (-0.7369959242164307, 10.697039838804978)]

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Lambda(lambda x: global_contrast_normalization(x)),
                                    transforms.Normalize([min_max[args.normal_class][0]],
                                                         [min_max[args.normal_class][1] \
                                                          - min_max[args.normal_class][0]])])
    train = datasets.MNIST(root=data_dir, train=True, download=True)
    test = datasets.MNIST(root=data_dir, train=False, download=True)

    x_train = train.data
    y_train = train.targets

    # 학습에 사용하는 데이터에는 normal data만 포함함
    x_train = x_train[np.where(y_train == args.normal_class)]
    y_train = y_train[np.where(y_train == args.normal_class)]

    data_train = MNIST_Dataset(x_train, y_train, transform)
    train_loader = DataLoader(data_train, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)

    # 테스트를 위해서는 normal, anomal이 섞여 있음
    # normal은 0으로 anomal은 1로 변환함
    x_test = test.data
    y_test = test.targets
    y_test = np.where(y_test == args.normal_class, 0, 1)
    data_test = MNIST_Dataset(x_test, y_test, transform)
    test_loader = DataLoader(data_test, batch_size=args.batch_size,
                             shuffle=True, num_workers=0)
    return train_loader, test_loader


# for data transformation
def global_contrast_normalization(x):
    # 데이터를 대조시킴
    # 샘플당 모든 pixel에 대한 평균을 구함
    mean = torch.mean(x)
    x -= mean
    x_scale = torch.mean(torch.abs(x))
    x /= x_scale
    return x


class Deep_SVDD(nn.Module):
    def __init__(self, z_dim=32):
        super(Deep_SVDD, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

# data를 새롭게 representation하기 위한 AutoEncoder
class C_AutoEncoder(nn.Module):
    def __init__(self, z_dim=32):
        super(C_AutoEncoder, self).__init__()
        self.z_dim = z_dim
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def encoder(self, x):
        # encoder 구조는 Deep SVDD와 완전히 동일한 구조를 가지고 있음
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        return self.fc1(x)

    def decoder(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2)
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
        x = self.deconv3(x)
        return torch.sigmoid(x)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

class TrainerDeepSVDD:
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device

    def pretrain(self):
        # Deep SVDD에 적용할 가중치 W를 학습하기 위해 autoencoder를 학습함
        ae = C_AutoEncoder(self.args.latent_dim).to(self.device)
        ae.apply(weights_init_normal)
        optimizer = optim.Adam(ae.parameters(), lr=self.args.lr_ae,
                               weight_decay=self.args.weight_decay_ae)

        # 지정한 step마다 learning rate를 줄여감
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.args.lr_milestones, gamma=0.1)

        # AE 학습
        ae.train()
        for epoch in range(self.args.num_epochs_ae):
            total_loss = 0
            for x, _ in (self.train_loader):
                x = x.float().to(self.device)

                optimizer.zero_grad()
                x_hat = ae(x)
                # 재구축 오차를 최소화하는 방향으로 학습함
                # AE 모델을 통해 그 데이터를 잘 표현할 수 있는 common features를 찾는 것이 목표임
                reconst_loss = torch.mean(torch.sum((x_hat - x) ** 2, dim=tuple(range(1, x_hat.dim()))))
                reconst_loss.backward()
                optimizer.step()

                total_loss += reconst_loss.item()
            scheduler.step()
            print('Pretraining Autoencoder... Epoch: {}, Loss: {:.3f}'.format(
                epoch, total_loss / len(self.train_loader)))
        self.save_weights_for_DeepSVDD(ae, self.train_loader)

    def save_weights_for_DeepSVDD(self, model, dataloader):

        # AE의 encoder 구조의 가중치를 Deep SVDD에 초기화하기 위함임
        c = self.set_c(model, dataloader)
        net = network(self.args.latent_dim).to(self.device)
        state_dict = model.state_dict()
        # 구조가 맞는 부분만 가중치를 load함
        net.load_state_dict(state_dict, strict=False)
        torch.save({'center': c.cpu().data.numpy().tolist(),
                    'net_dict': net.state_dict()}, 'weights/pretrained_parameters.pth')

    def set_c(self, model, dataloader, eps=0.1):

        # 구의 중심점을 초기화함
        model.eval()
        z_ = []
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.float().to(self.device)
                z = model.encode(x)
                z_.append(z.detach())
        z_ = torch.cat(z_)
        c = torch.mean(z_, dim=0)
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps
        return c

    def train(self):

        # AE의 학습을 마치고 그 가중치를 적용한 Deep SVDD를 학습함
        net = Deep_SVDD().to(self.device)

        if self.args.pretrain == True:
            state_dict = torch.load('weights/pretrained_parameters.pth')
            net.load_state_dict(state_dict['net_dict'])
            c = torch.Tensor(state_dict['center']).to(self.device)
        else:
            # pretrain을 하지 않았을 경우 가중치를 초기화함
            net.apply(weights_init_normal)
            c = torch.randn(self.args.latent_dim).to(self.device)

        optimizer = optim.Adam(net.parameters(), lr=self.args.lr,
                               weight_decay=self.args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=self.args.lr_milestones, gamma=0.1)

        net.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _ in (self.train_loader):
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

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1 and classname != 'Conv':
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Linear") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
# 파라미터 지정
class Args:
    num_epochs = 150
    num_epochs_ae = 150
    patience = 50
    lr = 1e-4
    weight_decay = 0.5e-6
    weight_decay_ae = 0.5e-3
    lr_ae = 1e-4
    lr_milestones = [50]
    batch_size = 200
    pretrain = True
    latent_dim = 32
    normal_class = 1


args = Args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = mnist_loader(args)

deep_SVDD = TrainerDeepSVDD(args, data, device)

# AE pretrain
if args.pretrain:
    deep_SVDD.pretrain()

deep_SVDD.train()
def eval(net, c, dataloader, device):
   # ROC AUC score 계산

    scores = []
    labels = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=1)

            scores.append(score.detach().cpu())
            labels.append(y.cpu())
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
    return labels, scores

labels, scores = eval(deep_SVDD.net, deep_SVDD.c, data[1], device)