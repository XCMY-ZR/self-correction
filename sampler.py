import torch
import torch.nn as nn

from core.attacks.base import Attack,LabelMixin
from core.attacks.utils import batch_multiply
from core.attacks.utils import clamp,normalize_by_pnorm,clamp_by_pnorm,batch_clamp
from core.attacks import CWLoss

import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim


class Sampler():

    def __init__(self, predict, loss=None,grad_norm=None, clip_min=0., clip_max=1.,eps=8./255.,eps_iter=2./255.,Hnoise_eps=8./255.,Lnoise_eps=0.):
        super(Sampler,self).__init__()
        if loss=='pxy':
            self.E_func = self.pxy
        elif loss=='logsumexp':
            self.E_func = self.losumexp
        elif loss=='ce':
            self.E_func = self.cross_enerty
        elif loss=='cw':
            self.E_func =self.cwloss
        else:
            raise Exception("check loss if in ['pxy','logsumexp','ce','cw']")
        self.model = predict
        self.grad_norm = grad_norm
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.eps = eps
        self.eps_iter = eps_iter
        self.Hnoise_eps = Hnoise_eps
        self.Lnoise_eps = Lnoise_eps
    
    def perturb(self,x,y):
        xadv = x.requires_grad_()
        logits = self.model(xadv)

        # pxy = torch.gather(logits, 1, y[:, None]).sum()
        E_score= self.E_func(logits,y)
        E_score.backward()
        
        if self.grad_norm == 'linf':
            grad = xadv.grad.detach().sign()
        elif self.grad_norm == 'l2':
            grad = xadv.grad.detach()
            grad = normalize_by_pnorm(grad)
        elif self.grad_norm == 'no':
            # assert self.grad_norm is None
            grad = xadv.grad.detach()
        else:
            raise Exception("check grad_norm if in ['linf','l2','no]")

        noise = torch.randn_like(xadv).to(xadv.device)

        xadv = xadv + batch_multiply(self.eps, grad) + batch_multiply(self.Lnoise_eps,noise)
        xadv = clamp(xadv, self.clip_min, self.clip_max)
        radv = xadv - x
        return xadv.detach(), radv.detach()

    def sample_Langevin(self,x,y,n_iter=10):
        delta = torch.zeros_like(x)
        delta.requires_grad_()

        for i in range(n_iter):
            logits = self.model(x+delta)
            E_score= self.E_func(logits,y)
            E_score.backward()

            if self.grad_norm == 'linf':
                grad_sign = delta.grad.data.sign()
                delta.data = delta.data + batch_multiply(self.eps_iter, grad_sign)
                delta.data = batch_clamp(self.eps, delta.data)
                delta.data = clamp(x.data + delta.data, self.clip_min, self.clip_max) - x.data
            elif self.grad_norm == 'l2':
                grad = delta.grad.data
                grad = normalize_by_pnorm(grad)
                delta.data = delta.data + batch_multiply(self.eps_iter, grad)
                delta.data = clamp(x.data + delta.data, self.clip_min, self.clip_max) - x.data
                if self.eps is not None:
                    delta.data = clamp_by_pnorm(delta.data, 2, self.eps)
            elif self.grad_norm == 'no':
                grad = delta.grad.data
                delta.data = delta.data+batch_multiply(self.eps, grad)
            else:
                raise Exception("check grad_norm if in ['linf','l2','no]")
            noise = torch.randn_like(x).to(x.device)
            delta.data += batch_multiply(self.Lnoise_eps,noise)
            delta.grad.data.zero_()
        x_adv = clamp(x + delta, self.clip_min, self.clip_max)
        r_adv = x_adv - x
        return x_adv, r_adv
        
    def sample_multi(self,x,y,n_iter=10):
        xadv=x
        for i in range(n_iter):
            xadv,radv= self.perturb(xadv,y) 
        return xadv.detach(), radv.detach()

    def pxy(self,logits,y):
        return torch.gather(logits, 1, y[:, None]).sum()
    
    def losumexp(self,logits,y):
        return logits.logsumexp(1).sum()
    
    def cross_enerty(self,logits,y):
        return -1.*F.cross_entropy(logits,y,reduction='sum')
    
    def cwloss(self,logists,y):
        return -1.*CWLoss(logists,y)
