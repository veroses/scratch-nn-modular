import numpy as np
from .layers import *
from abc import ABC, abstractmethod

##TODO: implement RMSProp

'''
params is a dict of parameters (weights and biases)
grads is a corresponding dict of gradients
'''


class Optimizer(ABC):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, params, grads):
        pass
 
class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = {}

    def update(self, params, grads, layer_idx):
        for p_key in params:
            key = f"{layer_idx}_{p_key}"
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[p_key])

            self.velocity[key] = self.momentum * self.velocity[key] - self.learning_rate * grads[p_key]
            params[p_key] += self.velocity[key]

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moment = {}
        self.velocity = {}
        self.t = 0


    def update(self, params, grads, layer_idx):
        self.t += 1
        for p_key in params:
            key = f"{layer_idx}_{p_key}"
            if key not in self.moment:
                self.moment[key] = np.zeros_like(params[p_key])
                self.velocity[key] = np.zeros_like(params[p_key])

            self.moment[key] = self.beta1 * self.moment[key] + (1 -  self.beta1) * grads[p_key]
            self.velocity[key] = self.beta2 * self.velocity[key] + (1 - self.beta2) * np.square(grads[p_key])

            moment_corr = self.moment[key] / (1 - self.beta1 ** self.t)
            velocity_corr = self.velocity[key] / (1 - self.beta2 ** self.t)

            params[p_key] -= self.learning_rate * moment_corr / (np.sqrt(velocity_corr) + self.epsilon)  


def get_optimizer(name, **kwargs):
    if name.lower() == "sgd":
        return SGD(**kwargs)
    elif name.lower() == "adam":
        return Adam(**kwargs)
    else:
        raise ValueError("unknown optimizer")
    
