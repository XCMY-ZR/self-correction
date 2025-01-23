import os
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision import datasets, transforms


from core.attacks import create_attack
from core.attacks import CWLoss, DLRloss,ATTACKS
from core.utils import seed
from core.utils import str2bool, str2float
from robustbench.utils import load_model

from AdvCorrecting import self_correcting_eval
#模型
from models import ResNet18


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--data',default='cifar10',type=str)
    parser.add_argument('--data-dir', default='../data', type=str)
    parser.add_argument('--model-name',default='ResNet18',type=str)
    # parser.add_argument('--model-file',type=str,required=True)
    parser.add_argument('--csv-file', type=str, help='Output directory')
    parser.add_argument('--seed', default=1, type=int, help='Random seed')
    #for self correcting
    # parser.add_argument('--s-loss',action='store_true',default=False)
    parser.add_argument('--s-loss-name',type=str,default='ce')
    parser.add_argument('--topk',type=int, default=10)
    parser.add_argument('--sampler_iters',type=int,default=10)
    parser.add_argument('--grad-norm',type=str,default='linf')
    parser.add_argument('--eps',default=8/255,type=str2float)
    # parser.add_argument('--noise-eps',default=0.01,type=float)
    #for attack
    parser.add_argument('--attack',type=str,choices=ATTACKS,default='linf-pgd',help="Type of attack")
    parser.add_argument('--attack-loss',type=str,choices=['cw','ce','dlr'],default='cw',help='loss use for attacking')
    parser.add_argument('--attack-eps', type=str2float, default=8/255, help='Epsilon for the attack.')
    parser.add_argument('--attack-step', type=str2float, default=2/255, help='Step size for PGD attack.')
    parser.add_argument('--attack-iter', type=int, default=40, help='Max. number of iterations (if any) for the attack.')
    # parser.add_argument('--ATmethods', default='PGDAT', type=str)
    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.log('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

def save_adversarial_dataset(adversarial_examples, labels, filename):
    # 将对抗样本和标签保存为字典
    dataset = {
        'data': adversarial_examples,
        'labels': labels
    }
    torch.save(dataset, filename)

class AdversarialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def cycle(loader):
    while True:
        for data in loader:
            yield data

def main():
    args = get_args()
    print_args(args)
    seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    #for load data
    advdatafile = './adv-data/{}-{}-{}.pth'.format(args.data, args.model_name, args.attack+args.attack_loss)
    if os.path.exists(advdatafile):
        # 加载保存的对抗样本数据集
        loaded_dataset = torch.load(advdatafile)
        custom_dataset = AdversarialDataset(loaded_dataset['data'], loaded_dataset['labels'])
        advloader = torch.utils.data.DataLoader(custom_dataset, batch_size=args.batch_size, shuffle=False,num_workers=10)
        advloader = cycle(advloader)
    else:
        advloader = None

    if args.data == 'cifar10':
        transformer_test  = transforms.Compose([transforms.ToTensor()])

        data_loader  = torch.utils.data.DataLoader(datasets.CIFAR10('../data/', train=False,
                                                                transform=transformer_test, download=True),
                                            batch_size=args.batch_size, shuffle=False, num_workers=10)
        
    elif args.data == 'cifar100':
        transformer_test  = transforms.Compose([transforms.ToTensor()])

        data_loader  = torch.utils.data.DataLoader(datasets.CIFAR100('../data/', train=False,
                                                                transform=transformer_test, download=True),
                                            batch_size=args.batch_size, shuffle=False, num_workers=10)
        
    elif args.data == 'imagenet':
        transformer_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        ds = datasets.ImageFolder(root=f"../data/ILSVRC2012/val", transform=transformer_test)
        data_loader = torch.utils.data.DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=10)

    else:
        raise Exception('please check you args.data name')
    
    #for load model
    if args.model_name=='ResNet18':
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)
        mu = torch.tensor(cifar10_mean).view(3,1,1).to(device)
        std = torch.tensor(cifar10_std).view(3,1,1).to(device)

        def with_norm(X):
            return (X - mu)/std

        model_test = ResNet18()
        checkpoint = torch.load('./checkpoints/resnet.pth')
        model_test = nn.DataParallel(model_test).to(device) # put this line after loading state_dict if the weights are saved without module.
        if 'state_dict' in checkpoint.keys():
            model_test.load_state_dict(checkpoint['state_dict'])
        else:
            model_test.load_state_dict(checkpoint)
        
        model_test = model_test.module
        class normalize_model(nn.Module):
            def __init__(self, model):
                super(normalize_model, self).__init__()
                self.model_test = model
            def forward(self, x):
                return self.model_test(with_norm(x))
        model = normalize_model(model_test)
        del checkpoint

    elif args.model_name=="Debenedetti2022Light_XCiT-S12":
        model = load_model(model_name=args.model_name, dataset=args.data, threat_model='Linf').to(device)
    elif args.model_name=='Debenedetti2022Light_XCiT-M12':
        model = load_model(model_name=args.model_name, dataset=args.data, threat_model='Linf').to(device)
    else:
        model = load_model(model_name=args.model_name,model_dir='/share/home/yyman/models',dataset=args.data, threat_model='Linf').to(device)
    model.eval()

    #for attack
    LOSS= {
        'ce': nn.CrossEntropyLoss(reduction="sum"),
        'cw': CWLoss,
        'dlr': DLRloss,
    }

    if args.attack == 'linf-apgd':
        eval_attack = create_attack(model, args.attack_loss, attack_type=args.attack, attack_eps=args.attack_eps, 
                                attack_iter=args.attack_iter, attack_step=args.attack_step)
        
    else:
        # gattack = create_attack(model, CWLoss, 'linf-pgd', 8/255, 40, 2/255)
        eval_attack = create_attack(model, LOSS[args.attack_loss], attack_type=args.attack, attack_eps=args.attack_eps, 
                                attack_iter=args.attack_iter, attack_step=args.attack_step)
    

    sce = self_correcting_eval(model=model,
                                s_loss=args.s_loss_name,
                                attack=eval_attack,
                                device=device,
                                topk=args.topk,
                                sampler_iters=args.sampler_iters,
                                eps=args.eps,)
    
    adv_data,labels=sce.evalall(dataloader=data_loader,
                                csv_file='./csv-out/{}-{}-{}-S{}-eval.csv'.format(args.data,args.model_name,args.attack+args.attack_loss+str(args.attack_iter),args.s_loss_name+str(args.sampler_iters)),
                                advloader=advloader)

    print(sce.agg_result)
    torch.save(sce.agg_result,'./csv-out/{}-{}-{}-S{}-agg.pth'.format(args.data,args.model_name,args.attack+args.attack_loss+str(args.attack_iter),args.s_loss_name+str(args.sampler_iters)))

    if not os.path.exists(advdatafile):
        print("save adv-data", advdatafile)
        save_adversarial_dataset(adv_data,labels,filename=advdatafile)
    

if __name__ == '__main__':
    main()
