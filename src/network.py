import random
import numpy as np

class LeNet5:
    def __init__(self, l2=0):
        self.layers = [Convolution(1, 6, (5, 5), (2, 2)), Relu(), Pooling((2, 2)), Convolution(6, 16, (5, 5)), Pooling((2, 2)), Flatten(), Linear(400, 120), Relu(), Linear(120, 84), Relu(), Linear(84, 369), SoftMaxCrossEntropy()]

    def feedforward(self, X):
        z = X
        for layer in self.layers[:-1]:
            z = layer.forward(z)

        a = softmax(z)

        return a

    def SGD(self, training_data, mini_batch_size, epochs, learning_rate, test_data=None):
        if test_data:
            test_size = len(test_data)

        training_size = len(training_data)
        
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[i: i + mini_batch_size] for i in range(0, training_size, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update(mini_batch, learning_rate)

            if test_data:
                print(f"Epoch {epoch}: {self.evaluate(test_data)} / {test_size}")

            else:
                print(f"Epoch {epoch} complete")

    def update(self, mini_batch, learning_rate):
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
            
            if isinstance(layer, Linear):
                layer.weights = layer.weights - learning_rate * layer.grad_w
                layer.biases = layer.biases - learning_rate * layer.grad_b

            elif isinstance(layer, Convolution):
                layer.kernel = layer.kernel - learning_rate * layer.grad_K
                layer.bias = layer.bias - learning_rate * np.sum(layer.grad_b, axis=(1, 2))


    def evaluate(self, test_data):
        X = np.stack([x for x, y in test_data])
        Y = np.stack([y for x, y in test_data])
        outputs = self.feedforward(X)
        predicted_labels = np.argmax(outputs, axis=1)
        true_labels = np.argmax(Y, axis=1)
        accuracy = np.mean(predicted_labels == true_labels)
        return accuracy
    
    def visualize_cost(self):
        return 

class Convolution:
    def __init__(self, in_channels, num_filters, ker_size, padding=(0,0), stride=(1, 1)):
        self.kernel = np.random.randn(num_filters, in_channels, ker_size[0], ker_size[1]) * np.sqrt(2 / ker_size[0])
        self.bias = np.zeros(num_filters)
        self.padding = padding
        self.stride = stride

    def forward(self, X):
        self.X = X
        return self.multi_out_cross_correlate(X, self.kernel) + self.bias.reshape(1, -1, 1, 1)
    
    def backward(self, delta_out):
        self.grad_K = np.sum(self.multi_out_cross_correlate(self.X, delta_out)) / delta_out.shape[0]
        self.grad_b = np.sum(delta_out, axis=0) / delta_out.shape[0]

        k_rotate = np.flip(self.kernel, (-1, -2))
        delta_in = np.sum(self.multi_out_cross_correlate(delta_out, k_rotate, mode="full"))
        return delta_in
    
    def backward_but_better(self, delta_out): #delta_out.shape = (B, F, output_height, output_width)\

        

        F, ker_channels, ker_height, ker_width = self.kernel.shape
        #reshape delta_out
        B, F, dout_height, dout_width = delta_out.shape
        delta_out = np.reshape(delta_out,( B, F, dout_height * dout_width)) 

        #reshape kernel into k_col
        kernel_matrix = self.kernel.reshape(F, -1)

        #perform matrix multiplication for dX_col and dK_col
        dX_col = np.matmul(kernel_matrix.T, delta_out)
        dK_col = np.matmul(delta_out, self.im_matrix.T)

        #reshape and calculate grad_B
        self.grad_K = np.sum(dK_col, axis=0).reshape(self.kernel.shape) / B
        self.grad_b = np.sum(delta_out, axis=0) / B
        delta_in = col2im(dX_col, self.X, self.kernel.shape, self.stride, self.padding)
        

        return delta_in

    def im2col_cross_corr(self, X, K, mode="reg"):
        pad_h, pad_w = self.padding
        if mode == "reg":
            X = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
            F, ker_channels, ker_height, ker_width = K.shape
            B, C, im_height, im_width = X.shape
            stride_h, stride_w = self.stride

            output_height = (im_height - ker_height + stride_h) // stride_h
            output_width = (im_width - ker_width + stride_w) // stride_w

            self.im_matrix = im2col(X, (ker_height, ker_width), self.stride, (output_height, output_width))

            kernel_matrix = K.reshape(F, -1)

            output = np.matmul(kernel_matrix[None, :, :], self.im_matrix)
            
            return output.reshape(B, F, output_height, output_width)

    def multi_out_cross_correlate(self, X, K, mode="reg"):
        return np.stack([np.stack([self.multi_in_cross_correlate(x, k, mode) for k in K]) for x in X])

    def multi_in_cross_correlate(self, X, K, mode="reg"):
        return np.sum([self.cross_correlate(x, k, mode) for x, k in zip(X, K)], axis=0)

    def cross_correlate(self, X, K, mode="reg"):
        ker_height, ker_width = K.shape
        stride_h, stride_w = self.stride
        

        if mode == "reg":
            X = np.pad(X, self.padding)
            x_height, x_width = X.shape
            H = np.zeros(((x_height - ker_height + stride_h) // stride_h,( x_width - ker_width + stride_w) // stride_w))

            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    H[i][j] = np.sum( X[i * stride_h: i * stride_h + ker_height, j * stride_w: j * stride_w + ker_width] * K)
            
        elif mode == "full":
            x_height, x_width = X.shape
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

    def im2col_pool(self, X, channel, sample):
        pad_h, pad_w = self.padding
        self.X = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
        B, channels, x_height, x_width = X.shape
        stride_v, stride_h = self.stride
        output_height = (x_height - self.pool_height + stride_v) // stride_v
        output_width = (x_width - self.pool_width + stride_h) // stride_h

        im_matrix = im2col(X, self.pool_size, self.stride, (output_height, output_width), flatten=False)

        if self.type == "max":
            pooled = np.max(im_matrix, axis=(-2, -1))
        else: 
            pooled = np.mean(im_matrix, axis=(-2, -1))

        return pooled 
    
    def pool(self, X, channel, sample):
        X = np.pad(X, self.padding)
        x_height, x_width = X.shape
        stride_v, stride_h = self.stride
        

        H = np.zeros(((x_height - self.pool_height + stride_v) // stride_v,( x_width - self.pool_width + stride_h) // stride_h))

        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                if self.type == "max":
                    max_val = np.max(X[i * stride_v: i * stride_v + self.pool_height, j * stride_v: j* stride_h + self.pool_width])
                    H[i][j] = max_val
                    rows, cols = np.where(X[i * stride_v: i * stride_v + self.pool_height, j * stride_h: j* stride_h + self.pool_width] == max_val)
                    global_row = stride_v * i + rows[0] - self.padding[0]
                    global_col = stride_h * j + cols[0] - self.padding[1]
                    self.route.append((sample, channel, global_row, global_col))
                else:
                    H[i][j] = np.mean(X[i * stride_v: i * stride_v + self.pool_height, j * stride_h: j* stride_h + self.pool_width])

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
        self.biases = np.zeros(out_dim,)
        self.weights = np.random.randn(out_dim, in_dim) * np.sqrt(2 / in_dim)

    def forward(self, x):
        self.x = x
        return x @ self.weights.T + self.biases

    def backward(self, delta_out):
        self.grad_w = delta_out.T @ self.x / delta_out.shape[0]
        self.grad_b = np.sum(delta_out, axis=0) / delta_out.shape[0]
        return delta_out @ self.weights

class Relu:
    def forward(self, x):
        self.x = x
        return relu(x)
    
    def backward(self, delta_out):
        return delta_out * relu_prime(self.x)
    
    
class SoftMaxCrossEntropy:
    def forward(self, logits, labels): #for use in training only
        self.probs = softmax(logits)
        return cross_entropy(self.probs, labels)
    
    def backward(self, labels):
        return self.probs - labels / len(labels)

class Flatten:
    def forward(self, X):
        self.dim = X.shape
        batch_size = self.dim[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, delta_out):
        delta_in = delta_out.reshape(self.dim)
        return delta_in


def softmax(z_L):
        z_stable = z_L - np.max(z_L, axis=1, keepdims=True)
        exp_z = np.exp(z_stable)
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)

        sum_exp_z = np.where(sum_exp_z == 0, 1e-12, sum_exp_z)

        return exp_z / sum_exp_z

def relu(z):
    return np.maximum(0, z)


def relu_prime(z):
    return (z > 0).astype(float)
    

def cross_entropy(output_a, y):
    return -np.sum( y * np.log(output_a + 1e-12)) / y.shape[0]


def cross_entropy_delta(a, y):
    return a - y

def get_indices(X_shape, k_size, stride, output_size):
    B, channels, im_height, im_width = X_shape
    k_height, k_width = k_size
    stride_h, stride_w = stride
    output_height, output_width = output_size

    batch_idx = np.arange(B).reshape(B, 1, 1, 1, 1, 1)
    channel_idx = np.arange(channels).reshape(1, channels, 1, 1, 1, 1)

    x1_idx = np.arange(0, im_width - k_width + 1, stride_w)
    y1_idx = np.arange(0, im_height - k_height + 1, stride_h)

    x_offsets, y_offsets = np.meshgrid(np.arange(0, k_width), np.arange(0, k_height), indexing="ij")

    xf_idx = np.add.outer(x1_idx, x_offsets)
    yf_idx = np.add.outer(y1_idx, y_offsets)

    xf_idx = xf_idx.reshape(1, 1, 1, output_width, 1, k_width)
    yf_idx = yf_idx.reshape(1, 1, output_height, 1, k_height, 1)

    return batch_idx, channel_idx, xf_idx, yf_idx


def im2col(X, size, stride, output_size, flatten="True"):
    k_height, k_width = size
    B, channels, im_height, im_width = X.shape
    stride_h, stride_w = stride
    output_height, output_width = output_size

    batch_idx, channel_idx, xf_idx, yf_idx = get_indices(X.shape, size, stride, output_size)

    patches = X[batch_idx, channel_idx, yf_idx, xf_idx]
    
    if flatten:
        return patches.reshape(B, channels * k_height * k_width, output_height * output_width)
    else:   
        return patches
    

def col2im(dX_col, X, size, stride, padding):
    pad_h, pad_w = padding
    k_height, k_width = size
    B, channels, im_height, im_width = X.shape
    stride_h, stride_w = stride
    B, channels, im_height, im_width = X.shape
    output_height = (im_height - k_height + stride_h) // stride_h
    output_width = (im_width - k_width + stride_w) // stride_w

    im = np.zeros_like(X)

    batch_idx, channel_idx, xf_idx, yf_idx = get_indices(X.shape, size, (output_height, output_width))
    
    np.add.at(im, (batch_idx, channel_idx, xf_idx, yf_idx), dX_col)

    if pad_h == 0 and pad_w == 0:
        return im
    else:
        return im[:, :, pad_h:im.shape[2]-pad_h, pad_w:im.shape[3]-pad_w]



