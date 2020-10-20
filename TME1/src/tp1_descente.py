import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context


# Les données supervisées
x = torch.randn(50, 13)
y = torch.randn(50, 3)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3)
b = torch.randn(3)

epsilon = 0.05

mse_cntxt=Context()
linear_cntxt=Context()
writer = SummaryWriter()
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)
    f= Linear.forward(linear_cntxt,x,w,b)
    loss=MSE.forward(mse_cntxt,f,y)
    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
    print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)
    id= torch.eye(50,3)
    mse_grad=MSE.backward(mse_cntxt,id)
    _,grad_w,grad_b= Linear.backward(linear_cntxt,mse_grad[1])
    ##  TODO:  Mise à jour des paramètres du modèle
    b= b + epsilon *grad_b
    w= w + epsilon *grad_w 

