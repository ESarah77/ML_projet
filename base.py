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
        # cout = self.forward(y, yhat)
        # return torch.autograd.grad(inputs=yhat, outputs=cout, grad_outputs=torch.ones_like(cout))
        return -2*(y-yhat)

class Linear(Module):
    def __init__(self, input, output):
        super().__init__()
        self._parameters = {
            "weights" : np.random.rand(input, output), # Initialisation aléatoire de la matrice de poids/paramètres W (avant init param = None)
            "bias" : np.random.rand(output) # Initialisation aléatoire du vecteur de biais
        }
        self._gradient = {
            "weights" : np.zeros((input, output)), # Initialisation aléatoire de la matrice de poids/paramètres W (avant init param = None)
            "bias" : np.zeros((1, output)) # Initialisation aléatoire du vecteur de biais
        }

    def zero_grad(self):
        self._gradient["weights"] = np.zeros(self._gradient["weights"])
        self._gradient["bias"] = np.zeros(self._gradient["bias"])

    def forward(self, data):
        return data@self._parameters["weights"] + self._parameters["bias"]
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters["weights"] -= gradient_step*self._gradient["weights"]
        self._parameters["bias"] -= gradient_step*self._gradient["bias"]

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        # equation 1 : dL/dW, en fonction de input et delta 
        self._gradient["weights"] += delta@input
        self._gradient["bias"] += np.sum(delta, axis=0)
        

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        # equation 2 : dL/dX, en fonction de input et delta
        return delta@self._parameters["weights"]
    
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

