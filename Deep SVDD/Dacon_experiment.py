import sys
import easydict
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
sys.path.append("C:\\Users\\user\\Desktop\\python\\Deep SVDD")
import DeepSVDD_Dacon as DS_D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args_0 = easydict.EasyDict({
       'num_epochs':700,
       'num_epochs_ae':400,
       'lr':1e-6,
       'lr_ae':1e-3,
       'weight_decay':5e-6,
       'weight_decay_ae':5e-3,
       'lr_milestones':[50],
       'batch_size':64,
       'pretrain':True,
       'latent_dim':32,
       'normal_class':0
                })

def Train_deepsvdd(args, device, train_file, test_file, train_folder, test_folder):

    dataloader_train, dataloader_test = DS_D.get_data(train_file,test_file,train_folder,test_folder,args)

    deep_SVDD = DS_D.TrainerDeepSVDD(args, dataloader_train, device)

    if args.pretrain:
        deep_SVDD.pretrain()

    net, c = deep_SVDD.train()

    return net, c, dataloader_test

def eval(net, c, dataloader, device):
    """Testing the Deep SVDD model"""
    threshold = 0.5
    predictions = []
    net.eval()
    print('Testing...')
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.float().to(device)
            # x = global_contrast_normalization(x)
            z = net(x)
            score = torch.sum((z - c) ** 2, dim=1)
            predictions.extend((score > threshold).int().cpu().tolist())

    return predictions

def Experiment(args, device, repeat, train_file, test_file, train_folder, test_folder):
    ''':param
    repeat : Repetition of experiment '''
    for i in range(repeat):
        print(str(i) + "실험")
        deepsvdd_net, deepsvdd_c, dataloader_test = Train_deepsvdd(args, device, train_file, test_file, train_folder, test_folder)
        predictions = eval(deepsvdd_net, deepsvdd_c, dataloader_test, device)
        submission_df = pd.read_csv('C:\\Users\\user\\Desktop\\open\\sample_submission.csv')
        submission_df.iloc[:, 1] = predictions
        submission_df.to_csv('C:\\Users\\user\\Desktop\\open\\sample_submission.csv', index=False)

if __name__ == '__main__':
    train_file = 'C:\\Users\\user\\Desktop\\open\\train.csv'
    test_file =  'C:\\Users\\user\\Desktop\\open\\test.csv'
    train_folder = 'C:\\Users\\user\\Desktop\\open\\train'
    test_folder = 'C:\\Users\\user\\Desktop\\open\\test'
    repeat = 1
    Experiment(args_0, device, repeat, train_file, test_file, train_folder, test_folder)