import numpy as np
import torch

########################### FONCTIONS UTILES ###################################

def prevent_overflow(data): # Fonction à utiliser dès qu'une valeur est donnée à l'exponentielle
    # 200 = borne pour éviter l'overflow lorsqu'un très grand nombre est donné à l'exponentielle
    data_without_overflow = np.where(data > 200, 200, data)
    data_without_overflow = np.where(data_without_overflow < -200, -200, data_without_overflow)
    return data_without_overflow

def one_hot_encode(labels, num_classes):
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot

def accuracy_multiclasse(predictions, targets):
    predicted_classes = np.argmax(predictions, axis=-1)
    true_classes = np.argmax(targets, axis=-1)
    accuracy = np.mean(predicted_classes == true_classes)
    return accuracy

def accuracy_bin(predictions, targets):
    pred = np.where(predictions >= 0.5, 1, 0)
    return np.mean(pred == targets)

################################# CLASSES DE BASES ################################

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
        return ((y - yhat)**2).mean(axis=-1)

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
        return np.tanh(data.astype(np.float64))
        
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
        data_without_overflow = prevent_overflow(data)
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
            # print(output[:10])
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

        if yhat.shape[-1] == 1: # classes binaires
            accuracy = accuracy_bin(yhat, batch_y)
        else:
            accuracy = accuracy_multiclasse(yhat, batch_y)

        return loss.mean(), accuracy
    
def SGD(net, loss, X, Y, batch_size, n_iter, learning_rate):
    # Découpage en batch et apprentissage du réseau
    opt = Optim(net, loss, learning_rate)
    nb_samples = X.shape[0]
    loss_history = []
    accuracy_history = []

    for i in range(n_iter):
        # Shuffle the examples
        indices = np.arange(nb_samples)
        np.random.shuffle(indices)
        data_shuffled = X[indices]
        labels_shuffled = Y[indices]
        tmp_loss = [] # loss for each batch
        tmp_acc = []

        # Mini-batch training
        for index_start in range(0, nb_samples, batch_size):
            index_end = min(index_start + batch_size, nb_samples) # Traitement dans le cas où il y a moins de samples pour le dernier batch
            batch_x = data_shuffled[index_start:index_end]
            batch_y = labels_shuffled[index_start:index_end]

            loss_value, accuracy_value = opt.step(batch_x, batch_y)
            tmp_loss.append(loss_value)
            tmp_acc.append(accuracy_value)
        
        loss_history.append(np.mean(tmp_loss)) # mean of the loss of each batch
        accuracy_history.append(np.mean(tmp_acc))

    return loss_history, accuracy_history

######################### 4E PARTIE : Softmax, LogSoftmax, CELogSoftmax ###############################

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def zero_grad(self):
        ## Annule gradient / Réinitialiser à 0 le gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        data_without_overflow = X - np.max(X, axis=-1, keepdims=True)
        exp_data = np.exp(X - np.max(X, axis=-1, keepdims=True)) # soustraire par le max pour éviter les instabilités numériques
        return exp_data / np.sum(exp_data, axis=-1, keepdims=True) # probas normalisées

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        output = self.forward(input)
        return delta * output * (1 - output)


class LogSoftmax(Module):
    def __init__(self):
        super().__init__()

    def zero_grad(self):
        ## Annule gradient / Réinitialiser à 0 le gradient
        pass

    def forward(self, X):
        ## Calcule la passe forward
        max_data = np.max(X, axis=-1, keepdims=True)
        data_without_overflow = prevent_overflow(X - max_data)
        log_sum_exp = np.log(np.sum(np.exp(X - max_data), axis=-1, keepdims=True))
        return X - max_data - log_sum_exp

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        output = self.forward(input)
        data_without_overflow = prevent_overflow(output)
        e = np.exp(output)
        return delta * (1 - e / np.sum(e, axis=-1, keepdims=True))


class CE(Loss):
    def forward(self, y, yhat): # y : one-hot encoded
        return -np.sum(y * yhat, axis=-1) # y et yhat matrix

    def backward(self, y, yhat):
        return yhat - y


class CELogSoftmax(Loss):
    def forward(self, y, yhat):
        data_without_overflow = prevent_overflow(yhat)
        return np.log(np.sum(np.exp(data_without_overflow), axis=1) + 1e-10) - np.sum(y * data_without_overflow, axis=1) 

    def backward(self, y, yhat):
        data_without_overflow = prevent_overflow(yhat)
        e = np.exp(data_without_overflow)
        return e / (np.sum(e, axis=1).reshape((-1, 1)) + 1e-10) - y
    
