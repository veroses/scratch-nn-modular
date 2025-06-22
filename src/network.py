import random
import numpy as np


class Convolution:
    def __init__(self, in_channels, num_filters, ker_size, padding=(0,0), stride=1):
        self.kernel = np.random.randn(num_filters, in_channels, ker_size[0], ker_size[1])
        self.bias = np.random.rand()
        self.padding = padding
        self.stride = stride

    def forward(self, X):
        return self.multi_out_cross_correlate(X, self.kernel, self.padding, self.stride) + self.bias
    
    def backward(self, delta):
        return
    
    def multi_out_cross_correlate(self, X, K, padding, stride):
        return np.stack([np.stack([self.multi_in_cross_correlate(x, k, padding, stride) for k in K]) for x in X])

    def multi_in_cross_correlate(self, X, K, padding, stride):
        return np.sum(self.cross_correlate(x, k, padding, stride) for x, k in zip(X, K))

    def cross_correlate(self, X, K, padding=(0,0), stride=1):

        X = np.pad(X, padding)

        ker_height, ker_width = K.shape
        x_height, x_width = X.shape

        H = np.zeros(((x_height - ker_height + padding[0] + stride) // stride,( x_width - ker_width + padding[1] + stride) // stride))

        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                H[i][j] = np.sum( X[i * stride: i * stride + ker_height, j * stride: j * stride + ker_width] * K)
        
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
                    global_col = stride_h * j + row + cols[1] - self.padding[1]
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
        self.biases = np.random.rand(out_dim, 1)
        self.weights = np.random.randn(out_dim, in_dim)

    def forward(self, x):
        self.x = x
        return x @ self.weights + self.biases

    def backward(self, grad_out):
        self.grad_w = grad_out.T @ self.x
        self.grad_b = np.sum(grad_out, axis=0) / len(grad_out[0])
        return grad_out @ self.weights


    

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
'''
class Network:
    def __init__(self, layer_sizes, l2_lambda=0.0, momentum_coe = 0.0):
        self.num_layers = len(layer_sizes)
        self.biases = [np.random.rand(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2 / x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.l2_lambda = l2_lambda
        self.momentum_coe = momentum_coe
        

    def feedforward(self, a): #calculate output of network given input a
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            a = relu(np.dot(weight, a) + bias)

        z_L = np.dot(self.weights[-1], a) + self.biases[-1]
        a = self.softmax(z_L)
        return a
    
    def softmax(self, z_L):
        z_stable = z_L - np.max(z_L) #avoid overflow with large z
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z)
    
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None): #stochastic gradient descent given mini batch size and number of epochs
        if test_data:
            n_testdata = len(test_data)

        velocity_w = [np.zeros_like(w) for w in self.weights]
        velocity_b = [np.zeros_like(b) for b in self.biases]

        training_size = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, training_size, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self.update(mini_batch, mini_batch_size, learning_rate, velocity_w, velocity_b)

            if test_data:
                print (f"Epoch {epoch}: {self.evaluate(test_data)} / {n_testdata}")

            else:
                print (f"Epoch {epoch} complete")


    def update(self, mini_batch, batch_size, learning_rate, velocity_w, velocity_b):

        X = np.hstack([x for x, y in mini_batch])
        Y = np.hstack([y for x, y in mini_batch])

        grad_w, grad_b = self.backprop(X, Y)
 
        grad_w = [g_w + self.l2_lambda * w / batch_size for g_w, w in zip(grad_w, self.weights)]

        for l in range(len(self.weights)):
            velocity_w[l] = self.momentum_coe * velocity_w[l] - learning_rate * grad_w[l]
            velocity_b[l] = self.momentum_coe * velocity_b[l] - learning_rate * grad_b[l]

            self.weights[l] = self.weights[l] + velocity_w[l]
            self.biases[l] = self.biases[l] + velocity_b[l]
         

    
    def backprop(self, X, Y): #backpropagation algorithm
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        activation = X
        activations = [X]
        z_values = []

        #run forward feed
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = np.matmul(weight, activation) + bias
            activation = relu(z)
            z_values.append(z)
            activations.append(activation)
        
        z_L = self.weights[-1] @ activation + self.biases[-1]
        z_values.append(z_L)
        activations.append(self.softmax(z_L))
        
        #start backwards run
        batch_size = X.shape[1]
        delta = [np.zeros((b.shape[0], batch_size)) for b in self.biases]

        for layer in reversed(range(len(self.weights))):
            if layer == self.num_layers - 2:
                delta[layer] = cross_entropy_delta(activations[-1], Y)
            else:
                delta[layer] = np.multiply(np.matmul(self.weights[layer + 1].T, delta[layer + 1]), relu_prime(z_values[layer]))
            grad_w[layer] = np.matmul(delta[layer], np.transpose(activations[layer])) / batch_size
            grad_b[layer] = np.mean(delta[layer], axis = 1, keepdims=True)
        return grad_w, grad_b
    
    def evaluate(self, test_data): #evaluate accuracy based on validation set
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in test_data]
        return sum(int(predicted == actual) for (predicted, actual) in test_results)
    
    def l2_penalty(self, n):
        return 0.5 * self.l2_lambda / n * sum(np.sum(w**2) for w in self.weights)
    


'''
def softmax(self, z_L):
        z_stable = z_L - np.max(z_L) #avoid overflow with large z
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z)

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)
    
def cross_entropy(output_a, y):
    return y * np.log(output_a) + (1 - y) * np.log(1 - output_a)

def cross_entropy_delta(a, y):
    return a - y