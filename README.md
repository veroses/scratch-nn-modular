# scratch\_nn

> Minimal, modular neural network library from scratch.

## Quick Start

```python
from scratch_nn.layers import Linear, Relu
from scratch_nn.network import Network
from scratch_nn.optimization import SGD
import numpy as np

layers = [
    Linear(2, 4),
    Relu(),
    Linear(4, 2)
]

net = Network(layers, optimizer=SGD, learning_rate=0.1)

train_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
train_Y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=float)
train_data = list(zip(train_X, train_Y))

net.train(train_data, mini_batch_size=2, epochs=100)

accuracy = net.evaluate(train_data, batch_size=2)
print(f"Training accuracy: {accuracy}/{len(train_data)}")
```

---

## API Overview

### Layers

Layers (`scratch_nn.layers`)

`Linear(in_dim, out_dim)` — Fully connected layer

`Relu()` — ReLU activation

`SoftMax()` — Softmax activation for training

`BatchNormFC(channels)` — Batch normalization for fully connected input

`BatchNormConv(channels)` — Batch normalization for convolutional input

`Convolution(in_channels, num_filters, ker_size, padding, stride)` — Convolutional layer

`Pooling(pool_size, type='max', padding, stride)` — Pooling layer

`Flatten()` — Flatten tensor to 2D

Optimizers (`scratch_nn.optimization`)

`SGD(learning_rate, momentum)`

`Adam(learning_rate, beta1, beta2, epsilon)`

Network (`scratch_nn.network`)

`Network(layers, optimizer, **optimizer_kwargs)` — Construct training pipeline

`.feedforward(X)` — Run forward pass

`.train(training_data, mini_batch_size, epochs, test_data=None)` — Full training loop

`.update(mini_batch)` — Single mini-batch backpropagation + parameter update

`.evaluate(test_data, batch_size=32)` — Compute count of correct predictions

`.visualize_cost()` — Return a matplotlib Figure of the loss curve (not yet implemented)


## Run Tests

```bash
pytest --maxfail=1 -q
```

---

