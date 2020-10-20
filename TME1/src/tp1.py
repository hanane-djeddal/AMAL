# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
import numpy as np
from torch.autograd import Function
from torch.autograd import gradcheck

###### Pytorch : @ : prod mat 
# X.T transposé
#X.shape
#len(X)
#X.sum ; X.sum(dim=0)

class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        #  TODO:  Renvoyer la valeur de la fonction
        return (1/(len(y)))*((yhat-y)**2).sum()

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        return ((2*grad_output/len(y))*(yhat-y), (-2*grad_output/len(y))*(yhat-y))
    
mse = MSE.apply
#  TODO:  Implémenter la fonction Linear(X, W, b)

class Linear(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, x, w,b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(x, w,b)
        #  TODO:  Renvoyer la valeur de la fonction
        return x @ w + b

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        x, w,_= ctx.saved_tensors
        return (grad_output @ (w.T) , (grad_output.T @ x).T, grad_output.sum(0))  
linear= Linear.apply