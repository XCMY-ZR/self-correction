import torch
import torch.nn as nn
import torch.nn.functional as F
from core.attacks.utils import batch_multiply
from core.attacks.utils import clamp,normalize_by_pnorm,clamp_by_pnorm,batch_clamp
from torch.autograd import Variable
import torch.optim as optim
from copy import deepcopy

def accuracy(true, preds):
    """
    Computes multi-class accuracy.
    Arguments:
        true (torch.Tensor): true labels.
        preds (torch.Tensor): predicted labels.
    Returns:
        Multi-class accuracy.
    """
    accuracy = (torch.softmax(preds, dim=1).argmax(dim=1) == true).sum().float()
    return accuracy.item()

class CModel(nn.Module):
    def __init__(self, model,topk=5,n_iter=10,eps=8./255.,eps_iter=2./255.,clip_min=0.,clip_max=1.):
        super(CModel, self).__init__()
        self.model = model
        self.topk = topk
        self.n_iter = n_iter
        self.eps = eps
        self.eps_iter =eps_iter
        self.clip_min=clip_min
        self.clip_max=clip_max
        self.model.eval()
        self.res=[0 for i in range(self.topk+1)]
    
    def E_func(self,logits,y):
        return -1.*F.cross_entropy(logits,y,reduction='sum')
    
    def sample(self,x,y):
        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(self.n_iter):
            logits = self.model(x+delta)
            E_score= self.E_func(logits,y)

            E_score.backward(retain_graph=True)
            # E_score.backward()
            grad_sign = delta.grad.data.sign()
            delta.data = delta.data + batch_multiply(self.eps_iter, grad_sign)
            delta.data = batch_clamp(self.eps, delta.data)
            delta.data = clamp(x.data + delta.data, self.clip_min, self.clip_max) - x.data

            delta.grad.data.zero_()
        x_adv = clamp(x + delta, self.clip_min, self.clip_max)
        r_adv = x_adv - x
        return x_adv, r_adv

    def forward(self,x):
        y_logits =self.model(x)
        xs = x.size()
        cn = y_logits.size()[1]

        x_repeat = x.view(xs[0], 1, xs[1], xs[2], xs[3]).repeat(1, self.topk, 1, 1, 1)
        x_repeat = x_repeat.view(xs[0] * self.topk, xs[1], xs[2], xs[3])    

        sample_y_target = y_logits.topk(self.topk)[1].view(-1)
        
        x_sample, x_delta_sample = self.sample(x_repeat,sample_y_target)

        y_sample_pre = self.model(x_sample)
        
        yy_prob = torch.max(y_sample_pre.softmax(dim=1),dim=1)[0]
        tmp = yy_prob.view(xs[0],self.topk)
        yyw = tmp/tmp.sum(dim=1).view(-1,1)
        aggd = yyw.view(-1).view(-1,1,1,1)*x_delta_sample
        aggd =  aggd.view(xs[0],-1, xs[1], xs[2], xs[3]).sum(dim=1)
        return self.model(x+aggd)
        # return res

             