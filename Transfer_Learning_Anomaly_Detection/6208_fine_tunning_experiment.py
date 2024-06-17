import pandas as pd
import torch
import easydict
from sklearn.metrics import roc_auc_score
import sys
sys.path.append("C:\\Users\\gyuhee\\PycharmProjects\\pythonProject\\Code\\dsvdd_TL")
import dsvdd_TL as DS
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args_0 = easydict.EasyDict({
       'num_epochs':30,
       'num_epochs_ae3':30,
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


def load_model(args, device,test_loader):
    deep_SVDD = DS.TrainerDeepSVDD(args, None, device)
    net = DS.DeepSVDD_network(args.latent_dim).to(device)

    # 저장된 가중치와 c값 불러오기
    state_dict = torch.load('C:\\Users\\gyuhee\\PycharmProjects\\pythonProject\\model_save\\best_acc_parameters.pth')
    net.load_state_dict(state_dict['net_dict'])
    c = torch.Tensor(state_dict['center']).to(device)

    deep_SVDD.net = net
    deep_SVDD.c = c

    #deep_SVDD.fine_tune(test_loader)
    return deep_SVDD

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

def Experiment(args,device, repeat, train_n, test_n):
    ''':param
    repeat : Repetition of experiment '''
    Result = []


    data_dir = "C:\\Users\\gyuhee\\Desktop\\연구실\\dfdsf"
    DS.dataloader_train, dataloader_test = DS.get_data(train_n, args, data_dir)
    print(f"Number of training samples: {len(DS.dataloader_train.dataset)}")
    print(f"Number of testing samples: {len(dataloader_test.dataset)}")

    for _ in range(repeat):
        trainer = DS.TrainerDeepSVDD(args, DS.dataloader_train, device)
        deepsvdd_net, deepsvdd_c = trainer.train()
        labels, scores, roc_auc_value = eval(deepsvdd_net, deepsvdd_c, dataloader_test, device)
        Result.append(roc_auc_value)


    # Result = pd.DataFrame(Result)
    # Result.to_csv("svdd_result_" + ".csv")
    return Result


repeat = 10
# 실험 실행

results = pd.DataFrame()
for n in range(10, 81, 10):
    result = Experiment(args_0, device, repeat, n, n*2)
    result_df = pd.DataFrame(result, columns=[f"n={n}"])
    results = pd.concat([results, result_df], axis=1)


results.to_csv("svdd_all_results.csv")