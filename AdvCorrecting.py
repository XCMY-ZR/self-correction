import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from core.utils import ctx_noparamgrad_and_eval
from copy import deepcopy
# from core.metrics import accuracy


from sampler import Sampler

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

class self_correcting_eval(Sampler):
    def __init__(self,model,s_loss,attack,device,topk=10,sampler_iters=10,grad_norm='linf', clip_min=0., clip_max=1.,eps=8./255.,eps_iter=2./255.,Hnoise_eps=8./255.,Lnoise_eps=0.01):
        super(self_correcting_eval,self).__init__(model, s_loss, grad_norm, clip_min, clip_max, eps, eps_iter,Hnoise_eps,Lnoise_eps)

        self.attack = attack
        self.device = device
        self.topk = topk
        self.s_iter = sampler_iters
        self.agg_result={
                         'cN':[0 for i in range(topk+1)],
                         'aN':[0 for i in range(topk+1)],
                        }
    
    def agg_sample_x_res(self,x,l,d,y,flag):
        '''
        x: dataset data x [b,c,h,n]
        l: dataset labels l [b]
        d: sample delta [b*topk,c,h,n]
        y: model predict logits for sample x [b*topk,class_nums]
        '''
        xs = x.size()#[b,c,h,n]
        cn = y.size()[1] #class num
        for k in range(2,self.topk+1):
            #choose
            dd , yy = deepcopy(d.detach()), deepcopy(y.detach())
            dd =  (dd.view(xs[0],-1, xs[1], xs[2], xs[3])[:,:k]).contiguous() #[b,k,c,h,n]
            dd = dd.view(-1, xs[1], xs[2], xs[3]) #[b*k,c,h,n]
            yy = (yy.view(xs[0],-1,cn)[:,:k]).contiguous() #[b,k,cn]
            yy = yy.view(-1,cn)#[b*k,cn]

            # y_pred = torch.max(y_logits,dim=1)[1]
            yy_prob = torch.max(yy.softmax(dim=1),dim=1)[0] #[b*k]
            tmp = yy_prob.view(xs[0],k) #[b,k]
            yyw = tmp/tmp.sum(dim=1).view(-1,1)
            aggd = yyw.view(-1).view(-1,1,1,1)*dd
            aggd = aggd.view(xs[0],-1, xs[1], xs[2], xs[3]).sum(dim=1)

            with torch.no_grad():
                scy = self.model(x+aggd)
                self.agg_result[flag][k] += accuracy(l,scy)

    def each_ce(self,output,target):
        return torch.gather(output.logsumexp(dim=1,keepdim=True)-output, 1, target[:, None])

    def eval_Langevin(self,dataloader,csv_file):
        self.model.eval()
        clean_acc = 0
        adv_acc=0
 
        metrics = pd.DataFrame(columns=['ture_label', 'y_clean_label','y_clean_prob','y_adv_label','y_adv_prob','y_clean_sample_label','y_adv_sample_label',
                                'y_Langevin_clean_label','y_Langevin_clean_prob','y_Langevin_adv_label','y_Langevin_adv_prob'])
        metrics.to_csv(csv_file,index=False)

        for x , y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            with ctx_noparamgrad_and_eval(self.model):
                x_adv, _ = self.attack.perturb(x,y)
                # x_adv, _ = self.attack.perturb(x)
            with torch.no_grad():
                y_clean = self.model(x)
                y_adv = self.model(x_adv)

            clean_acc += accuracy(y,y_clean)
            print('cle x acc is {}'.format(clean_acc))

            adv_acc += accuracy(y,y_adv)
            print('adv x acc is {}'.format(adv_acc))

            # x_tmp = [x[i].re]
            xs = x.size()
            x_repeat = x.view(xs[0], 1, xs[1], xs[2], xs[3]).repeat(1, self.topk, 1, 1, 1)
            x_repeat = x_repeat.view(xs[0] * self.topk, xs[1], xs[2], xs[3])
            x_adv_repeat = x_adv.view(xs[0],1,xs[1],xs[2],xs[3]).repeat(1,self.topk,1,1,1)
            x_adv_repeat = x_adv_repeat.view(xs[0] * self.topk, xs[1], xs[2], xs[3])
            
            sample_clean_y=y_clean.topk(self.topk)[1].view(-1)
            sample_adv_y = y_adv.topk(self.topk)[1].view(-1)

            x_sampler_repeat_innorm, x_delta_repeat_innorm = self.sample_Langevin(x_repeat,sample_clean_y,self.s_iter)
            x_adv_sampler_repeat_innorm,x_adv_delta_repeat_innorm = self.sample_Langevin(x_adv_repeat,sample_adv_y,self.s_iter)
            
            # x_delta_innorm= torch.norm(x_delta_repeat_innorm.reshape([len(x_delta_repeat_innorm),-1]),p=2,dim=1)
            # x_adv_delta_innorm= torch.norm(x_adv_delta_repeat_innorm.reshape([len(x_delta_repeat_innorm),-1]),p=2,dim=1)

            with torch.no_grad():
                y_sampler_repeat_innorm= self.model(x_sampler_repeat_innorm)
                y_adv_sampler_repeat_innorm= self.model(x_adv_sampler_repeat_innorm)


            y_clean_pred = torch.max(y_clean,dim=1)[1]
            y_clean_prob = torch.max(y_clean.softmax(dim=1),dim=1)[0]

            y_adv_pred = torch.max(y_adv,dim=1)[1]
            y_adv_prob = torch.max(y_adv.softmax(dim=1),dim=1)[0]

            y_innorm_clean_pred = torch.max(y_sampler_repeat_innorm,dim=1)[1]
            y_innorm_clean_prob = torch.max(y_sampler_repeat_innorm.softmax(dim=1),dim=1)[0]


            y_innorm_adv_pred = torch.max(y_adv_sampler_repeat_innorm,dim=1)[1]
            y_innorm_adv_prob = torch.max(y_adv_sampler_repeat_innorm.softmax(dim=1),dim=1)[0]    

 

            batch_metrics = {'ture_label': torch.cat([i.repeat(self.topk) for i in y]).cpu().numpy(),
                         'y_clean_label': torch.cat([i.repeat(self.topk) for i in y_clean_pred]).cpu().numpy(),
                         'y_clean_prob': torch.cat([i.repeat(self.topk) for i in y_clean_prob]).cpu().numpy(),
                         'y_adv_label': torch.cat([i.repeat(self.topk) for i in y_adv_pred]).cpu().numpy(),
                         'y_adv_prob': torch.cat([i.repeat(self.topk) for i in y_adv_prob]).cpu().numpy(),
                         'y_clean_sample_label':sample_clean_y.cpu().numpy(),
                         'y_adv_sample_label':sample_adv_y.cpu().numpy(),
                         'y_Langevin_clean_label':y_innorm_clean_pred.cpu().numpy(),
                         'y_Langevin_clean_prob':y_innorm_clean_prob.cpu().numpy(),
                        #  'y_nonorm_clean_label':y_nonorm_clean_pred.cpu().numpy(),
                        #  'y_nonorm_clean_prob':y_nonorm_clean_prob.cpu().numpy(),
                        #  'x_delta_innorm':x_delta_innorm.detach().cpu().numpy(),
                        #  'x_delta_nonorm':x_delta_nonorm.detach().cpu().numpy(),
                         'y_Langevin_adv_label':y_innorm_adv_pred.cpu().numpy(),
                         'y_Langevin_adv_prob':y_innorm_adv_prob.cpu().numpy(),
                        #  'y_nonorm_adv_label':y_nonorm_adv_pred.cpu().numpy(),
                        #  'y_nonorm_adv_prob':y_nonorm_adv_prob.cpu().numpy(),
                        #  'x_adv_delta_innorm':x_adv_delta_innorm.detach().cpu().numpy(),
                        #  'x_adv_delta_nonorm':x_adv_delta_nonorm.detach().cpu().numpy(),
                         }
            new_data = pd.DataFrame(batch_metrics)

            # 追加到文件中，忽略列名
            new_data.to_csv(csv_file, mode='a', header=False, index=False)

    def evalall(self,dataloader,csv_file,advloader=None):
        self.model.eval()
        clean_acc = 0
        adv_acc=0
 
        metrics = pd.DataFrame(columns=['ture_label', 'y_clean_label','y_clean_prob','y_adv_label','y_adv_prob','y_clean_sample_label','y_adv_sample_label',
                                'y_Langevin_clean_label','y_Langevin_clean_prob','y_Langevin_adv_label','y_Langevin_adv_prob','ES_Langevin_clean','ES_Langevin_adv',
                                        ])
        metrics.to_csv(csv_file,index=False)

        adv_list = []
        labels_list = []

        for x , y in dataloader:
            x, y = x.to(self.device), y.to(self.device)
            if advloader:
                x_adv, yy = advloader.__next__()
                assert any(y.cpu()==yy)
                x_adv = x_adv.to(self.device)
            else:
                with ctx_noparamgrad_and_eval(self.model):
                    x_adv, _ = self.attack.perturb(x,y)
                    adv_list.append(x_adv.cpu())
                    labels_list.append(y.cpu())
                # x_adv, _ = self.attack.perturb(x)
            with torch.no_grad():
                y_clean = self.model(x)
                y_adv = self.model(x_adv)

            clean_acc += accuracy(y,y_clean)
            print('cle x acc is {}'.format(clean_acc))

            adv_acc += accuracy(y,y_adv)
            print('adv x acc is {}'.format(adv_acc))

            # x_tmp = [x[i].re]
            xs = x.size()
            x_repeat = x.view(xs[0], 1, xs[1], xs[2], xs[3]).repeat(1, self.topk, 1, 1, 1)
            x_repeat = x_repeat.view(xs[0] * self.topk, xs[1], xs[2], xs[3])
            # x_repeat = x_repeat + torch.randn_like(x_repeat) * 0.02
            x_adv_repeat = x_adv.view(xs[0],1,xs[1],xs[2],xs[3]).repeat(1,self.topk,1,1,1)
            x_adv_repeat = x_adv_repeat.view(xs[0] * self.topk, xs[1], xs[2], xs[3])
            # x_adv_repeat = x_adv_repeat + torch.randn_like(x_adv_repeat) * 0.02
            
            sample_clean_y=y_clean.topk(self.topk)[1].view(-1)
            sample_adv_y = y_adv.topk(self.topk)[1].view(-1)

            x_sampler_repeat_Langevin, x_delta_repeat_Langevin = self.sample_Langevin(x_repeat,sample_clean_y,self.s_iter)
            x_adv_sampler_repeat_Langevin,x_adv_delta_repeat_Langevin = self.sample_Langevin(x_adv_repeat,sample_adv_y,self.s_iter)
   

            with torch.no_grad():
                y_sampler_repeat_Langevin= self.model(x_sampler_repeat_Langevin)
                y_adv_sampler_repeat_Langevin= self.model(x_adv_sampler_repeat_Langevin)


            E_Score_langevin_clean =self.each_ce(y_sampler_repeat_Langevin, sample_clean_y)
            E_Score_langevin_adv =self.each_ce(y_adv_sampler_repeat_Langevin, sample_adv_y) 


            y_clean_pred = torch.max(y_clean,dim=1)[1]
            y_clean_prob = torch.max(y_clean.softmax(dim=1),dim=1)[0]

            y_adv_pred = torch.max(y_adv,dim=1)[1]
            y_adv_prob = torch.max(y_adv.softmax(dim=1),dim=1)[0]

            y_Langevin_clean_pred = torch.max(y_sampler_repeat_Langevin,dim=1)[1]
            y_Langevin_clean_prob = torch.max(y_sampler_repeat_Langevin.softmax(dim=1),dim=1)[0]
            y_Langevin_adv_pred = torch.max(y_adv_sampler_repeat_Langevin,dim=1)[1]
            y_Langevin_adv_prob = torch.max(y_adv_sampler_repeat_Langevin.softmax(dim=1),dim=1)[0]    


            batch_metrics = {'ture_label': torch.cat([i.repeat(self.topk) for i in y]).cpu().numpy(),
                         'y_clean_label': torch.cat([i.repeat(self.topk) for i in y_clean_pred]).cpu().numpy(),
                         'y_clean_prob': torch.cat([i.repeat(self.topk) for i in y_clean_prob]).cpu().numpy(),
                         'y_adv_label': torch.cat([i.repeat(self.topk) for i in y_adv_pred]).cpu().numpy(),
                         'y_adv_prob': torch.cat([i.repeat(self.topk) for i in y_adv_prob]).cpu().numpy(),
                         'y_clean_sample_label':sample_clean_y.cpu().numpy(),
                         'y_adv_sample_label':sample_adv_y.cpu().numpy(),
                         'y_Langevin_clean_label':y_Langevin_clean_pred.cpu().numpy(),
                         'y_Langevin_clean_prob':y_Langevin_clean_prob.cpu().numpy(),
                         'y_Langevin_adv_label':y_Langevin_adv_pred.cpu().numpy(),
                         'y_Langevin_adv_prob':y_Langevin_adv_prob.cpu().numpy(),
                         'ES_Langevin_clean':E_Score_langevin_clean.detach().cpu().squeeze(dim=-1).numpy(),
                         'ES_Langevin_adv':E_Score_langevin_adv.detach().cpu().squeeze(dim=-1).numpy(),
                         }
            new_data = pd.DataFrame(batch_metrics)
            # 追加到文件中，忽略列名
            new_data.to_csv(csv_file, mode='a', header=False, index=False)
            self.agg_sample_x_res(x=x,l=y,d=x_delta_repeat_Langevin,y=y_sampler_repeat_Langevin,flag='cN')
            self.agg_sample_x_res(x=x_adv,l=y,d=x_adv_delta_repeat_Langevin,y=y_adv_sampler_repeat_Langevin,flag='aN')

            print(self.agg_result)

        if advloader:
            return None,None
        else:
            adv_data = torch.cat(adv_list, dim=0)
            labels_data = torch.cat(labels_list, dim=0)
            return adv_data, labels_data
