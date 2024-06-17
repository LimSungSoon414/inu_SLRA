
import sys
import easydict
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
sys.path.append("C:\\Users\\user\\Desktop\\python\\Deep SVDD")
import Deep_SVDD_MNIST as DS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args_0 = easydict.EasyDict({
       'num_epochs':10,
       'num_epochs_ae':10,
       'lr':1e-5,
       'lr_ae':1e-4,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':200,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':0
                })

args_1 = easydict.EasyDict({
       'num_epochs':350,
       'num_epochs_ae':250,
       'lr':1e-5,
       'lr_ae':1e-4,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':200,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':1
                })

args_2 = easydict.EasyDict({
       'num_epochs':350,
       'num_epochs_ae':250,
       'lr':1e-5,
       'lr_ae':1e-4,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':200,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':2
                })

args_3 = easydict.EasyDict({
       'num_epochs':350,
       'num_epochs_ae':250,
       'lr':1e-5,
       'lr_ae':1e-4,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':200,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':3
                })

args_4 = easydict.EasyDict({
       'num_epochs':350,
       'num_epochs_ae':250,
       'lr':1e-5,
       'lr_ae':1e-4,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':200,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':4
                })

args_5 = easydict.EasyDict({
       'num_epochs':350,
       'num_epochs_ae':250,
       'lr':1e-5,
       'lr_ae':1e-4,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':200,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':5
                })

args_6 = easydict.EasyDict({
       'num_epochs':350,
       'num_epochs_ae':250,
       'lr':1e-5,
       'lr_ae':1e-4,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':200,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':6
                })

args_7 = easydict.EasyDict({
       'num_epochs':350,
       'num_epochs_ae':250,
       'lr':1e-5,
       'lr_ae':1e-4,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':200,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':7
                })

args_8 = easydict.EasyDict({
       'num_epochs':350,
       'num_epochs_ae':250,
       'lr':1e-5,
       'lr_ae':1e-4,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':200,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':8
                })

args_9 = easydict.EasyDict({
       'num_epochs':350,
       'num_epochs_ae':250,
       'lr':1e-5,
       'lr_ae':1e-4,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':200,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':9
                })
def Train_deepsvdd(args, device):

    if __name__ == '__main__':
        # Train/Test Loader 불러오기
        dataloader_train, dataloader_test = DS.get_mnist(args)

        # Network 학습준비, 구조 불러오기
        deep_SVDD = DS.TrainerDeepSVDD(args, dataloader_train, device)

        # DeepSVDD를 위한 DeepLearning pretrain 모델로 Weight 학습
        if args.pretrain:
            deep_SVDD.pretrain()

        # 학습된 가중치로 Deep_SVDD모델 Train
        net, c = deep_SVDD.train()

    return net, c, dataloader_test

def eval(net, c, dataloader, device):
    """Testing the Deep SVDD model"""
    scores = []
    labels = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for x, y in dataloader:
            x = x.float().to(device)
            # x = global_contrast_normalization(x)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=1)

            scores.append(score.detach().cpu())
            labels.append(y.cpu())
    labels, scores = torch.cat(labels).numpy(), torch.cat(scores).numpy()
    roc_auc_value = roc_auc_score(labels, scores)
    print(f'ROC AUC score: {roc_auc_value:.4f}')
    return labels, scores, roc_auc_value

def Experiment(args,device, repeat):
    ''':param
    repeat : Repetition of experiment '''
    Result = []
    for i in range(repeat):
        print(str(i) + "실험")
        deepsvdd_net, deepsvdd_c, dataloader_test = Train_deepsvdd(args,device)
        labels, scores, roc_auc_value = eval(deepsvdd_net, deepsvdd_c, dataloader_test, device)
        # # labels, scores = eval(deep_SVDD.net, deep_SVDD.c, data[1], device)
        #print(eval(net=deep_SVDD.net,c=deep_SVDD.c,dataloader=dataloader_test,device=device))
        Result.append(roc_auc_value)
    Result = pd.DataFrame(Result)
    Result.to_csv("MNIST_" + str(args.normal_class) + ".csv")

repeat = 10

Experiment(args_0,device, repeat)
Experiment(args_1,device, repeat)
Experiment(args_2,device, repeat)
Experiment(args_3,device, repeat)
Experiment(args_4,device, repeat)
Experiment(args_5,device, repeat)
Experiment(args_6,device, repeat)
Experiment(args_7,device, repeat)
Experiment(args_8,device, repeat)
Experiment(args_9,device, repeat)

