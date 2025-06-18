import random
import numpy as np



class Network:
    def __init__(self, layer_sizes, l2_lambda=0.0, momentum_coe = 0.0):
        self.num_layers = len(layer_sizes)
        self.biases = [np.random.rand(y, 1) for y in layer_sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(1 / x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
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
                velocity_w, velocity_b = self.update(mini_batch, mini_batch_size, learning_rate, velocity_w, velocity_b)

            if test_data:
                print (f"Epoch {epoch}: {self.evaluate(test_data)} / {n_testdata}")

            else:
                print (f"Epoch {epoch} complete")


    def update(self, mini_batch, mini_batch_size, learning_rate, velocity_w, velocity_b):

        X = np.hstack([x for x, y in mini_batch])
        Y = np.hstack([y for x, y in mini_batch])

        grad_w, grad_b = self.backprop(X, Y)

        grad_w = grad_w + self.l2_lambda * self.weights / mini_batch_size

        for l in range(len(self.weights)):
            velocity_w[l] = self.momentum_coe * velocity_w[l] - learning_rate * grad_w[l]
            velocity_b[l] = self.momentum_coe * velocity_b[l] - learning_rate * grad_b[l]

            self.weights[l] = self.weights[l] + velocity_w[l]
            self.biases[l] = self.biases[l] + velocity_b[l]
        return velocity_w, velocity_b

    
    def backprop(self, X, Y): #backpropagation algorithm
        grad_w = [np.zeros_like(w) for w in self.weights]
        grad_b = [np.zeros_like(b) for b in self.biases]
        activation = X
        activations = [X]
        z_values = []

        #run forward feed
        for weight, bias in zip(self.weights, self.biases):
            z = np.matmul(weight, activation) + bias
            activation = relu(z)

            z_values.append(z)
            activations.append(activation)
        
        #start backwards run
        batch_size = X.shape[1]
        delta = [np.zeros((b.shape[0], batch_size)) for b in self.biases]

        for layer in range(self.num_layers - 2, -1, -1):
            if layer == self.num_layers - 2:
                delta[-1] = cross_entropy_delta(activations[-1], Y)
            else:
                delta[layer] = np.multiply(np.matmul(self.weights[layer + 1].T, delta[layer + 1]), relu_prime(z_values[layer]))
            grad_w[layer] = np.matmul(delta[layer], np.transpose(activations[layer])) / batch_size
            grad_b[layer] = np.mean(delta[layer], axis = 1, keepdims=True)
        return grad_w, grad_b
    
    def evaluate(self, test_data): #evaluate accuracy based on validation set
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(predicted == actual) for (predicted, actual) in test_results)
    
    def l2_penalty(self, n):
        return 0.5 * self.l2_lambda / n * sum(np.sum(w**2) for w in self.weights)
    

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)
    
def cross_entropy(output_a, y):
    return y * np.log(output_a) + (1 - y) * np.log(1 - output_a)

def cross_entropy_delta(a, y):
    return a - y

