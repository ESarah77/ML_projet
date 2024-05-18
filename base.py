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
        ## Annule gradient / Réinitialiser à 0 le gradient
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
        return delta @ self._parameters["weights"].T
    
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
        return delta * (1 - self.forward(input)**2) 
    
class Sigmoide(Module):
    def __init__(self):
        super().__init__() # On ne récupère pas les paramètres car on n'en a pas besoin dans cette couche
    
    def zero_grad(self):
        pass

    def forward(self, data):
        # 709 = borne supérieure pour éviter l'overflow lorsqu'un très grand nombre est donné à l'exponentielle
        data_without_overflow = np.where(data > 709, 709, data)
        data_without_overflow = np.where(data_without_overflow < -709, -709, data_without_overflow)

        return 1 / (1 + np.exp(-data_without_overflow))
        
    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass # Ne fait rien car il n'y a pas de paramètres
    
    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass  # Ne fait rien car il n'y a pas de paramètres

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        output = self.forward(input)
        return delta * output * (1 - output)

######################### 3E PARTIE : Sequentiel et Optim ###############################

class Sequentiel(Module):
    # Classe qui permet de réaliser les étapes forward ou backward sur l'ensemble du réseau
    def __init__(self):
        super().__init__()
        self._modules = []
        self._outputs = [] # Enregistrer les outputs de chaque couche, car c'est les inputs utilisés pour le backward
    
    def zero_grad(self):
        for layer in self._modules:
            layer.zero_grad()
    
    def forward(self, data):
        input = data
        for layer in self._modules:
            output = layer.forward(input)
            self._outputs.append(output)
            input = output # Mise à jour des données d'entrée
        return output

    def update_parameters(self, gradient_step=1e-3):
        for layer in self._modules:
            layer.update_parameters(gradient_step)
    
    def backward_update_gradient(self, input, delta):
        pass

    def backward_delta(self, input, delta):
        pass

    def backward(self, input, delta):
        for i in range(len(self._modules)-1, 0, -1): # Parcours des modules de la fin vers le début
            layer = self._modules[i]
            input = self._outputs[i-1] # input = output de la couche précédente
            
            delta_next_turn = layer.backward_delta(input, delta)
            layer.backward_update_gradient(input, delta)
            delta = delta_next_turn
            if hasattr(layer, "input"): # Mise à jour des données d'entrée (seulement s'il y en a)
                input = layer.input
    
    def add_module(self, layer):
        self._modules.append(layer)


class Optim(object):
    # Permet de réaliser une itération de la descente de gradient sur l'ensemble du réseau
    def __init__(self, net, loss, eps):
        self._net = net
        self._loss = loss
        self._eps = eps
    
    def step(self, batch_x, batch_y):
        # Forward
        yhat = self._net.forward(batch_x)
        loss = self._loss.forward(batch_y, yhat)

        # Backward
        delta = self._loss.backward(batch_y, yhat)
        self._net.backward(yhat, delta)

        self._net.update_parameters(self._eps)

        return loss.mean()
    
def SGD(net, loss, X, Y, batch_size, n_iter, learning_rate, milestone):
    # Découpage en batch et apprentissage du réseau
    opt = Optim(net, loss, learning_rate)
    nb_samples = X.shape[0]
    loss_history = []

    for i in range(n_iter):
        # Shuffle the examples
        indices = np.arange(nb_samples)
        np.random.shuffle(indices)
        data_shuffled = X[indices]
        labels_shuffled = Y[indices]
        tmp_loss = [] # loss for each batch

        # Mini-batch training
        for index_start in range(0, nb_samples, batch_size):
            index_end = min(index_start + batch_size, nb_samples) # Traitement dans le cas où il y a moins de samples pour le dernier batch
            batch_x = data_shuffled[index_start:index_end]
            batch_y = labels_shuffled[index_start:index_end]

            loss_value = opt.step(batch_x, batch_y)
            tmp_loss.append(loss_value)
        
        # Keep a record of the loss every milestone epochs
        if i % milestone == 0:
            loss_history.append(np.mean(tmp_loss)) # mean of the loss of each batch

    return loss_history
