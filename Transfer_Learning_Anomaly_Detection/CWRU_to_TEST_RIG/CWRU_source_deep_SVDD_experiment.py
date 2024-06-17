import pandas as pd
import torch
import easydict
from sklearn.metrics import roc_auc_score
import sys
sys.path.append("/Transfer_Learning_Anomaly_Detection")
from Transfer_Learning_Anomaly_Detection.CWRU_to_TEST_RIG import CWRU_source_deep_SVDD as DS
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args_0 = easydict.EasyDict({
       'num_epochs':30,
       'num_epochs_ae':30,
       'lr':1e-3,
       'lr_ae':1e-3,
       'weight_decay':5e-7,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':32,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':0
                })

def Train_deepsvdd(args, device):
    if __name__ == '__main__':

        # Train/Test Loader 불러오기
        dataloader_train, dataloader_test = DS.get_data(args)

        # Network 학습준비, 구조 불러오기
        deep_SVDD = DS.TrainerDeepSVDD(args, dataloader_train, device)

        # DeepSVDD를 위한 DeepLearning pretrain 모델로 Weight 학습
        if args.pretrain:
            deep_SVDD.pretrain()

        # 학습된 가중치로 Deep_SVDD모델 Train
        net, c = deep_SVDD.train()
    return net,c,dataloader_test


# ##################
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

def Experiment(args,device, repeat):
    ''':param
    repeat : Repetition of experiment '''
    Result = []
    best_auc = 0.0
    best_model_weight = None  # 초기 최고 모델의 가중치
    best_c = None

    for i in range(repeat):
        print(str(i) + "실험")
        deepsvdd_net, deepsvdd_c, dataloader_test  = Train_deepsvdd(args,device)
        labels, scores, roc_auc_value = eval(deepsvdd_net, deepsvdd_c, dataloader_test, device)
        # # labels, scores = eval(deep_SVDD.net, deep_SVDD.c, data[1], device)
        #print(eval(net=deep_SVDD.net,c=deep_SVDD.c,dataloader=dataloader_test,device=device))
        print(f'ROC AUC score for experiment {i + 1}: {roc_auc_value:.2f}')
        Result.append(roc_auc_value)

        if roc_auc_value > best_auc:
            best_auc = roc_auc_value
            best_model_weight = copy.deepcopy(deepsvdd_net.state_dict())
            best_c = copy.deepcopy(deepsvdd_c)
            # best_r = copy.deepcopy(deepscdd_r)

    Result = pd.DataFrame(Result)
    Result.to_csv("svdd_result_" + ".csv")

    print(f'All ROC AUC scores: {Result}')

    # if best_model_weight is not None and best_c is not None:
    #     torch.save({'center': best_c.cpu().data.numpy().tolist(),
    #                 'net_dict': best_model_weight},'C:/Users/user/Desktop/python/Transfer_Learning_Anomaly_Detection/retrained_parameters_Testrig300.pth')

repeat = 10

Experiment(args_0,device, repeat)