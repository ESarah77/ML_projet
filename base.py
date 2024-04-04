# Page qui explique comment faire les dérivées partielles dans la chaine de règles
# https://medium.com/@rizqinur2010/partial-derivatives-chain-rule-using-torch-autograd-grad-a8b5917373fa
import numpy as np
import torch


class Loss(object):
    def forward(self, y, yhat):
        pass

    def backward(self, y, yhat):
        pass     

class Module(object):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        ## Annule gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

######################## 1ERE PARTIE : MSELoss et Linear ##############################
### A TESTER !!

class MSELoss(Loss):
    def forward(self, y, yhat):
        # y et yhat de taille batch x d
        return ((y - yhat)**2).mean(axis=1)

    def backward(self, y, yhat):
        cout = self.forward(y, yhat)
        return torch.autograd.grad(inputs=yhat, outputs=cout, grad_outputs=torch.ones_like(cout))

class Linear(Module):
    def __init__(self, input, output):
        super().__init__(_parameters, _gradient)
        self.input = input
        self.output = output # On n'en a pas besoin
        self._parameters = np.random.rand(input, output) # Initialisation aléatoire de la matrice de poids/paramètres W (avant init param = None)
        self.zero_grad() # Initialisation du gradient à 0 (avant init gradient = None)

    def zero_grad(self):
        self._gradient = 0
    
    def forward(self, X):
        return X@self._parameters
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters -= gradient_step*self._gradient

    def backward_update_gradient(self, input, delta):
        # equation 1 : dérivée du module/gradient du cout par rapport à ses paramètres, en fonction de input et delta 
        derivee = torch.autograd.grad(inputs=self._parameters, outputs=input, grad_outputs=delta)
        self._gradient += derivee

    def backward_delta(self, input, delta):
        # equation 2 : dérivée du module/gradient du cout par rapport à ses entrées, en fonction de input et delta
        derivee = torch.autograd.grad(inputs=self.input, outputs=input, grad_outputs=delta)
        return derivee
    
######################### 2E PARTIE : TanH et Sigmoide ###############################

class TanH(Module):
    def __init__(self, input, output):
        super().__init__(_gradient) # On ne récupère pas les paramètres car on n'en a pas besoin dans cette couche
        self.input = input
        self.output = output # On n'en a pas besoin
        self.zero_grad() # Initialisation du gradient à 0 (avant init gradient = None)

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        return # Ne fait rien car il n'y a pas de paramètres
    
    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass
    
class Sigmoide(Module):
    def __init__(self, input, output):
        super().__init__(_gradient) # On ne récupère pas les paramètres car on n'en a pas besoin dans cette couche
        self.input = input
        self.output = output # On n'en a pas besoin
        self.zero_grad() # Initialisation du gradient à 0 (avant init gradient = None)

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        return # Ne fait rien car il n'y a pas de paramètres
    
    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass

