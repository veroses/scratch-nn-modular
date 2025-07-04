import random
import numpy as np
from .layers import *
from .optimization import *
from .losses import *

class Network:
    def __init__(self, layers, optimizer, **kwargs):
        self.layers = layers
        self.optimizer = optimizer(**kwargs)
        self.losses = []

    def feedforward(self, X):
        z = X
        for layer in self.layers:
            z = layer.forward(z)

        return z

    def train(self, training_data, mini_batch_size, epochs, test_data=None):
        if test_data:
            test_size = len(test_data)

        training_size = len(training_data)

        for epoch in range(epochs):
            epoch_loss = 0
            random.shuffle(training_data)
            mini_batches = [training_data[k: k + mini_batch_size] for k in range(0, training_size, mini_batch_size)]

            for mini_batch in mini_batches:
                batch_loss = self.update(mini_batch)
                epoch_loss += batch_loss
            average_loss = epoch_loss / mini_batch_size
            self.losses.append(average_loss)
            
            if test_data:
                    print(f"Epoch {epoch}: {self.evaluate(test_data)} / {test_size}")
            else:
                print(f"Epoch {epoch} complete")


    def update(self, mini_batch):
        X = np.stack([x for x, y in mini_batch])
        Y = np.stack([y for x, y in mini_batch])
        Y = np.squeeze(Y, axis=-1)

        for  layer in self.layers:
            X = layer.forward(X)

        probs = X

        loss = cross_entropy(X, Y)

        delta = cross_entropy_delta(probs, Y)

        for layer in reversed(self.layers):
            delta = layer.backward(delta)

        for idx, layer in enumerate(self.layers):
            self.optimizer.update(layer.params, layer.grads, idx)

        return loss


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




