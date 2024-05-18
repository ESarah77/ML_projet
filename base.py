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

class MSELoss(Loss):
    def forward(self, y, yhat):
        # y et yhat de taille batch x d
        return ((y - yhat)**2).mean(axis=1)

    def backward(self, y, yhat):
        return -2*(y-yhat)

class Linear(Module):
    def __init__(self, input, output):
        super().__init__()
        self._parameters = {
            "weights" : np.random.rand(input, output), # Initialisation aléatoire de la matrice de poids/paramètres W (avant init param = None)
            "bias" : np.random.rand(output).reshape(1, -1) # Initialisation aléatoire du vecteur de biais
        }
        self._gradient = {
            "weights" : np.zeros((input, output)), # Initialisation des gradients des poids à 0
            "bias" : np.zeros((1, output)) # Initialisation des gradients de biais à 0
        }

    def zero_grad(self):
        self._gradient["weights"] = np.zeros(self._gradient["weights"].shape)
        self._gradient["bias"] = np.zeros(self._gradient["bias"].shape)

    def forward(self, data):
        return data@self._parameters["weights"] + self._parameters["bias"]
    
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters["weights"] -= gradient_step*self._gradient["weights"]
        self._parameters["bias"] -= gradient_step*self._gradient["bias"]

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        # equation 1 : dL/dW, en fonction de input et delta 
        self._gradient["weights"] += input.T @ delta
        self._gradient["bias"] += np.sum(delta, axis=0)
        
    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        # equation 2 : dL/dX, en fonction de input et delta
        return self._parameters["weights"] @ delta.T
    
######################### 2E PARTIE : TanH et Sigmoide ###############################

class TanH(Module):
    def __init__(self):
        super().__init__() # On ne récupère pas les paramètres car on n'en a pas besoin dans cette couche

    def zero_grad(self):
        pass

    def forward(self, data):
        return np.tanh(data)
        
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass # Ne fait rien car il n'y a pas de paramètres
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass  # Ne fait rien car il n'y a pas de paramètres

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        return (1 - self.forward(input)**2) * delta.T
    
class Sigmoide(Module):
    def __init__(self):
        super().__init__() # On ne récupère pas les paramètres car on n'en a pas besoin dans cette couche
    
    def zero_grad(self):
        pass

    def forward(self, data):
        return 1 / (1 + np.exp(-data))
        
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass # Ne fait rien car il n'y a pas de paramètres
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass  # Ne fait rien car il n'y a pas de paramètres

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        output = self.forward(input)
        return output * (1 - output) * delta

