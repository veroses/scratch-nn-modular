from layers import Layer
import numpy as np


class Relu(Layer):
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)
    
    def backward(self, delta_out):
        return delta_out * (self.X > 0).astype(float)
    

class SoftMax(Layer):
    def forward(self, logits): #for use in training only
        z_stable = logits - np.max(logits, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)

        sum_exp_z = np.where(sum_exp_z == 0, 1e-12, sum_exp_z)
        self.probs = exp_z / sum_exp_z
        return self.probs
    
    def backward(self, delta_out):
        return delta_out
    