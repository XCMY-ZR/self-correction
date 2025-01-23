import numpy as np

import torch
from autoattack.square import SquareAttack
from .base import LabelMixin


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# from .square import SquareAttack
# self.square = SquareAttack(self.model, p_init=.8, n_queries=5000, eps=self.epsilon, norm=self.norm,
#     n_restarts=1, seed=self.seed, verbose=False, device=self.device, resc_schedule=False)

class Square(LabelMixin):
    """
    APGD attack (from AutoAttack) (Croce et al, 2020).
    The attack performs nb_iter steps of adaptive size, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (str): loss function - ce or dlr.
        n_restarts (int): number of random restarts.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
        ord (int): (optional) the order of maximum distortion (inf or 2).
    """
    def __init__(self, predict, eps=0.3, ord=np.inf, n_restarts=1, seed=1, device=device):

        assert ord in [2, np.inf], 'Only ord=inf or ord=2 are supported!'
        self.targeted = False
        self.predict = predict
        norm = 'Linf' if ord == np.inf else 'L2'

        self.square = SquareAttack(predict, p_init=.8, n_queries=5000, eps=eps, norm=norm,
                                   n_restarts=1, seed=seed, verbose=False, device=device, resc_schedule=False)

    def perturb(self, x, y=None):
        x, y =self._verify_and_process_inputs(x,y)
        x_adv = self.square.perturb(x, y)
        r_adv = x_adv - x
        return x_adv, r_adv

    
class LinfSquareAttack(Square):
    """
    APGD attack (from AutoAttack) with order=Linf.
    The attack performs nb_iter steps of adaptive size, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (str): loss function - ce or dlr.
        n_restarts (int): number of random restarts.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
    """
    
    def __init__(self, predict, n_restarts=1, eps=8/255, seed=1, device=device):
        ord = np.inf
        super(LinfSquareAttack, self).__init__(
            predict=predict, eps=eps, ord=ord, n_restarts=n_restarts, seed=1, device=device)


class L2SquareAttack(Square):
    """
    APGD attack (from AutoAttack) with order=L2.
    The attack performs nb_iter steps of adaptive size, while always staying within eps from the initial point.
    Arguments:
        predict (nn.Module): forward pass function.
        loss_fn (str): loss function - ce or dlr.
        n_restarts (int): number of random restarts.
        eps (float): maximum distortion.
        nb_iter (int): number of iterations.
    """
    
    def __init__(self, predict, n_restarts=1, eps=8/255, seed=1, device=device):
        ord = 2
        super(L2SquareAttack, self).__init__(
            predict=predict, eps=eps, ord=ord, n_restarts=n_restarts, seed=1, device=device)

