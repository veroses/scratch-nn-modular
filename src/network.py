import random
import numpy as np
from layers import *
from optimization import *

class LeNet5:
    def __init__(self, optimizer, **kwargs):
        self.layers = [Convolution(1, 6, (5, 5), (2, 2)), Batch_NormConv(6), Relu(), Pooling((2, 2)), 
                       Convolution(6, 16, (5, 5)), Batch_NormConv(16), Relu(), Pooling((2, 2)), 
                       Flatten(), 
                       Linear(400, 120), Batch_NormFC(120), Relu(), 
                       Linear(120, 84), Batch_NormFC(84), Relu(), 
                       Linear(84, 10), SoftMaxCrossEntropy()]
        self.key = {}
        self.params = {}
        self.grads = {}

        self.optimizer = get_optimizer(optimizer, **kwargs)

        l_count = 0
        c_count = 0
        bc_count = 0
        bfc_count = 0
        for layer in self.layers:
            if isinstance(layer, Linear):
                l_count += 1
                self.key[layer] = (f"lin{l_count}_weights", f"lin{l_count}_biases")
                
                weights_key, biases_key = self.key[layer]

                self.params[weights_key] = layer.weights
                self.params[biases_key] = layer.biases
                self.grads[weights_key] = np.zeros_like(layer.weights)
                self.grads[biases_key] = np.zeros_like(layer.biases)

            elif isinstance(layer, Convolution):
                c_count += 1
                self.key[layer] = (f"conv{c_count}_weights", f"con{c_count}_biases")
                weights_key, biases_key = self.key[layer]
                self.params[weights_key] = layer.kernel
                self.params[biases_key] = layer.biases
                self.grads[weights_key] = np.zeros_like(layer.kernel)
                self.grads[biases_key] = np.zeros_like(layer.biases)

            elif isinstance(layer, Batch_NormConv):
                bc_count += 1
                self.key[layer] = (f"batchnorm_conv{bc_count}_weights", f"batchnorm_conv{bc_count}_biases")
                weights_key, biases_key = self.key[layer]
                self.params[weights_key] = layer.gamma
                self.params[biases_key] = layer.beta
                self.grads[weights_key] = np.zeros_like(layer.gamma)
                self.grads[biases_key] = np.zeros_like(layer.beta)

            elif isinstance(layer, Batch_NormFC):
                bfc_count += 1
                self.key[layer] = (f"batchnorm_fc{bfc_count}_weights", f"batchnorm_fc{bfc_count}_biases")
                weights_key, biases_key = self.key[layer]
                self.params[weights_key] = layer.gamma
                self.params[biases_key] = layer.beta
                self.grads[weights_key] = np.zeros_like(layer.gamma)
                self.grads[biases_key] = np.zeros_like(layer.beta)
        for key in self.params:
            print(key, self.params[key].dtype)

    def feedforward(self, X):
        z = X
        for layer in self.layers[:-1]:
            z = layer.forward(z)

        a = softmax(z)

        return a

    def train(self, training_data, mini_batch_size, epochs, test_data=None):
        if test_data:
            test_size = len(test_data)

        training_size = len(training_data)

        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, training_size, mini_batch_size)]

            for mini_batch in mini_batches:
                #start = time.time()
                self.update(mini_batch)
                #print("one minibatch: ", time.time() - start)
            print(f"epoch {epoch} done, waiting for evaluate")
            if test_data:
                    print(f"Epoch {epoch}: {self.evaluate(test_data)} / {test_size}")
            else:
                print(f"Epoch {epoch} complete")


    def update(self, mini_batch):
        X = np.stack([x for x, y in mini_batch])
        Y = np.stack([y for x, y in mini_batch])
        Y = np.squeeze(Y, axis=-1)
        z = X

        for layer in self.layers[:-1]:
            z = layer.forward(z)

        z = self.layers[-1].forward(z, Y)
        delta = self.layers[-1].backward(Y)

        for layer in reversed(self.layers[:-1]):
            delta = layer.backward(delta)

        self.collect_grads()
        #start = time.time()
        self.optimizer.update(self.params, self.grads)
        #print("Optimizer update time:", time.time() - start)



    def collect_grads(self):
        for layer in self.layers:
            if layer in self.key:
                    weights_key, biases_key = self.key[layer]
                    self.grads[weights_key] = layer.grad_w
                    self.grads[biases_key] = layer.grad_b

    def evaluate(self, test_data, batch_size=32):
        mini_batches = [test_data[k: k + batch_size] for k in range(0, len(test_data), batch_size)]
        accuracy = 0
        for batch in mini_batches:
            X = np.stack([x for x, y in batch])
            Y = np.stack([y for x, y in batch])
            outputs = self.feedforward(X)
            predicted_labels = np.argmax(outputs, axis=1)
            true_labels = np.argmax(Y, axis=1).flatten()
            accuracy += np.sum(predicted_labels == true_labels)
        return accuracy
    
    def visualize_cost(self):
        return 




