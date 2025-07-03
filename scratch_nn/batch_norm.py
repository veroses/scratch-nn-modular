from .layers import Layer
import numpy as np

#TODO: add mode switching

class BatchNormFC(Layer):
    def __init__(self, channels, epsilon=1e-8):
        self.gamma = np.ones(channels)
        self.beta = np.ones(channels)
        self.epsilon = epsilon

        self.params = {"g": self.gamma, "b": self.beta}
        self.grads = {"g": np.zeros_like(self.gamma), "b": np.zeros_like(self.beta)}

    def forward(self, X):
        self.X = X

        self.mean = np.mean(X, axis=0)
        self.variance = np.var(X, axis=0)
        self.X_center = self.X - self.mean
        self.X_norm = self.X_center / np.sqrt(self.variance + self.epsilon)

        out = self.X_norm * self.gamma + self.beta

        return out

    def backward(self, delta_out): #shape (B, C)
        M, C = delta_out.shape
        self.grads["g"] = np.sum(delta_out * self.X_norm, axis=0)
        self.grads["b"] = np.sum(delta_out, axis=0)

        std_inv = 1. / np.sqrt(self.variance + self.epsilon)
        factor = -0.5 * self.gamma * ( self.variance + self.epsilon) ** (-3/2)

        delta_x_norm = delta_out * self.gamma
        delta_v = np.sum(delta_out * self.X_center * factor, axis=0)
        delta_m = np.sum(delta_out * (-self.gamma * std_inv), axis=0) + delta_v * np.mean(-2 * self.X_center, axis=0)

        delta_in = delta_x_norm * std_inv + delta_v * 2 * self.X_center / M + delta_m / M
        return delta_in
    
class BatchNormConv:
    def __init__(self, channels, epsilon=1e-8):
        self.gamma = np.ones((1, channels, 1, 1))
        self.beta = np.ones((1, channels, 1, 1))
        self.epsilon = epsilon

        self.params = {"g": self.gamma, "b": self.beta}
        self.grads = {"g": np.zeros_like(self.gamma), "b": np.zeros_like(self.beta)}

    def forward(self, X):
        self.X = X

        self.mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
        self.variance = np.var(X, axis=(0, 2, 3), keepdims=True)
        self.X_center = self.X - self.mean
        self.X_norm = self.X_center / np.sqrt(self.variance + self.epsilon)

        out = self.X_norm * self.gamma + self.beta
        return out

    def backward(self, delta_out):
        B, C, H, W = delta_out.shape
        self.grads["g"] = np.sum(delta_out * self.X_norm, axis=(0, 2, 3), keepdims=True)
        self.grads["b"] = np.sum(delta_out, axis=(0, 2, 3), keepdims=True)

        std_inv = 1. / np.sqrt(self.variance + self.epsilon)
        factor = -0.5 * self.gamma * ( self.variance + self.epsilon) ** (-3/2)

        delta_x_norm = delta_out * self.gamma
        delta_v = np.sum(delta_out * self.X_center * factor, axis=(0, 2, 3), keepdims=True)
        delta_m = np.sum(delta_out * (-self.gamma * std_inv), axis=(0, 2, 3), keepdims=True) + delta_v * np.mean(-2 * (self.X_center), axis=(0, 2, 3), keepdims=True)

        delta_in = delta_x_norm * std_inv + delta_v * 2 * self.X_center / (B * H * W) + delta_m / (B * H * W)
        return delta_in