import random
import numpy as np

class LeNet5:
    def __init__(self):
        self.layers = [Convolution(1, 6, (5, 5), (2, 2)), Relu(), Pooling((2, 2)), Convolution(6, 16, (5, 5)), ]

class Convolution:
    def __init__(self, in_channels, num_filters, ker_size, padding=(0,0), stride=1):
        self.kernel = np.random.randn(num_filters, in_channels, ker_size[0], ker_size[1])
        self.bias = np.random.rand((num_filters,))
        self.padding = padding
        self.stride = stride

    def forward(self, X):
        self.X = X
        return self.multi_out_cross_correlate(X, self.kernel) + self.bias
    
    def backward(self, delta_out):
        self.grad_K = np.sum(self.multi_out_cross_correlate(self.X, delta_out), axis=0) / delta_out.shape[0]
        self.grad_b = np.sum(delta_out, axis=0) / delta_out.shape[0]

        k_rotate = np.flip(self.kernel, (-1, -2))
        delta_in = np.sum(self.multi_out_cross_correlate(delta_out, k_rotate, mode="full"), axis=0)
        return delta_in
    
    def multi_out_cross_correlate(self, X, K, mode="reg"):
        return np.stack([np.stack([self.multi_in_cross_correlate(x, k, mode) for k in K]) for x in X])

    def multi_in_cross_correlate(self, X, K, mode="reg"):
        return np.sum(self.cross_correlate(x, k, mode) for x, k in zip(X, K))

    def cross_correlate(self, X, K, mode="reg"):

        ker_height, ker_width = K.shape
        x_height, x_width = X.shape

        if mode == "reg":
            X = np.pad(X, self.padding)
            H = np.zeros(((x_height - ker_height + self.padding[0] + self.stride) // self.stride,( x_width - ker_width + self.padding[1] + self.stride) // self.stride))

            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    H[i][j] = np.sum( X[i * self.stride: i * self.stride + ker_height, j * self.stride: j * self.stride + ker_width] * K)
            
        elif mode == "full":
            
            H = np.zeros((x_height + ker_height - 1, x_width + ker_width - 1))

            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    for r in range(ker_height):
                        for c in range(ker_width):
                            x_j = j - c
                            x_i = i - r
                            if 0 <= x_i < x_height and 0 <= x_j < x_width:
                                H[i][j] += K[r][c] * X[x_i][x_j]
                    
        return H


class Pooling:
    def __init__(self, pool_size, type="max", padding=(0,0), stride=None):
        self.pool_size = pool_size
        self.pool_height, self.pool_width = pool_size
        self.type = type
        self.padding = padding
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride

    def forward(self, X):
        self.X = X
        self.route = []
        return np.stack([self.multi_in_pool(X[i], i) for i in range(len(X))])
        
    def multi_in_pool(self, X, sample):
            return np.stack([self.pool(X[j], j, sample) for j in range(len(X))])

    def pool(self, X, channel, sample):
        X = np.pad(X, self.padding)

        x_height, x_width = X.shape
        stride_v, stride_h = self.stride

        H = np.zeros(((x_height - self.pool_height + stride_h) // self.stride,( x_width - self.pool_width + stride_v) // self.stride))

        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if self.type == "max":
                    max = np.max(X[i * self.stride: i * self.stride + self.pool_height, j * self.stride: j* self.stride + self.pool_width])
                    H[i][j] = max
                    rows, cols = np.where(X[i * self.stride: i * self.stride + self.pool_height, j * self.stride: j* self.stride + self.pool_width] == max)
                    global_row = stride_v * i + rows[0] - self.padding[0]
                    global_col = stride_h * j + cols[0] - self.padding[1]
                    self.route.append((sample, channel, global_row, global_col))
                else:
                    H[i][j] = np.mean(X[i * self.stride: i * self.stride + self.pool_height, j * self.stride: j* self.stride + self.pool_width])

        return H
    
    def backward(self, delta_out):
        delta_in = np.zeros_like(self.X)
        if self.type == "max":
            indices = np.array(self.route).T
            delta_vals = delta_out.reshape(-1) #flatten
            np.add.at(delta_in, (indices[0], indices[1], indices[2], indices[3]), delta_vals)
        else:
            delta_in = np.stack([self.mean_back_pool(y) for y in delta_out])
        return delta_in
        
    def mean_back_pool(self, Y):
        return np.stack([self.mean_back_pool2D(y) for y in Y])

    def mean_back_pool2D(self, y):
        delta = np.repeat(np.repeat(y, self.pool_height, axis=0), self.pool_width, axis=1) / (self.pool_height * self.pool_width)
        return delta

    
class Linear:
    def __init__(self, in_dim, out_dim):
        self.biases = np.random.rand(out_dim,)
        self.weights = np.random.randn(out_dim, in_dim)

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.biases

    def backward(self, delta_out):
        self.grad_w = self.x.T @ delta_out / delta_out.shape[0]
        self.grad_b = np.sum(delta_out, axis=0) / delta_out.shape[0]
        return delta_out @ self.weights

class Relu:
    def forward(self, x):
        self.x = x
        return relu(x)
    
    def backward(self, delta_out):
        return delta_out * relu_prime(self.x)
    
    
class SoftMaxCrossEntropy:
    def __init__(self, logits, labels):
        #logits (batch_size, outputs)
        self.logits = logits
        self.labels = labels

    def forward(self, logits, labels): #for use in training only
        self.probs = softmax(logits)
        return cross_entropy(self.probs, labels)
    
    def backward(self):
        return self.probs - self.labels

class Flatten:
    def forward(self, X):
        self.dim = X.shape
        batch_size = self.dim[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, delta_out):
        delta_in = delta_out.reshape(self.dim)
        return delta_in


def softmax(z_L):
        z_stable = z_L - np.max(z_L) #avoid overflow with large z
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    return (z > 0).astype(float)
    

def cross_entropy(output_a, y):
    return -np.sum( y * np.log(output_a)) / y.shape[0]


def cross_entropy_delta(a, y):
    return a - y