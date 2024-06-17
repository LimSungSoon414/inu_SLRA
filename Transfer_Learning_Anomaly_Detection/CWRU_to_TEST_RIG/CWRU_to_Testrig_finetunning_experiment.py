import pandas as pd
import torch
import easydict
from sklearn.metrics import roc_auc_score
import sys
import copy
sys.path.append("/Transfer_Learning_Anomaly_Detection\\CWRU_to_TEST_RIG")
import CWRU_to_TestRig_finetunning as DS

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args_0 = easydict.EasyDict({
       'num_epochs':30,
       'num_epochs_ae':30,
       'lr':1e-3,
       'lr_ae':1e-3,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':5,
       'pretrain':False,
       'z_dim':32,
       'normal_class':0
                })

def load_pretrained_parameters(net, path_to_parameters):
    checkpoint = torch.load(path_to_parameters)
    net.load_state_dict(checkpoint['net_dict'])
    c = checkpoint['center']
    return net, torch.tensor(c, device=device)

def Train_deepsvdd(args, device, path_to_parameters=None):
    if __name__ == '__main__':
        # Train/Test Loader 불러오기
        dataloader_train, dataloader_test = DS.get_data(args)

        # Network 학습준비, 구조 불러오기
        deep_SVDD = DS.TrainerDeepSVDD(args, dataloader_train, device)

        # 저장된 파라미터가 있을 경우 불러오기
        if path_to_parameters is not None:
            deep_SVDD.net, deep_SVDD.c = load_pretrained_parameters(deep_SVDD.net, path_to_parameters)

        # 학습된 가중치로 Deep_SVDD모델 Train
        deep_SVDD.train()

    return deep_SVDD.net, deep_SVDD.c, dataloader_test

def eval(net, c, dataloader, device):
    """Testing the Deep SVDD model"""
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
    roc_auc_value = roc_auc_score(labels, scores)
    print('ROC AUC score: {:.2f}'.format(roc_auc_score(labels, scores)*100))
    return labels, scores, roc_auc_value

def Experiment(args, device, repeat, path_to_parameters=None):
    Result = []

    for i in range(repeat):
        print(f"{i}번째 실험")
        # Deep SVDD 모델 학습
        deepsvdd_net, deepsvdd_c, dataloader_test = Train_deepsvdd(args, device, path_to_parameters)
        labels, scores, roc_auc_value = eval(deepsvdd_net, deepsvdd_c, dataloader_test, device)
        print(f'ROC AUC score for experiment {i + 1}: {roc_auc_value:.2f}')
        Result.append(roc_auc_value)

    Result = pd.DataFrame(Result, columns=['ROC AUC'])
    Result.to_csv("svdd_result.csv", index=False)

    print(f'All ROC AUC scores: \n{Result}')

repeat = 10
Experiment(args_0, device, repeat)