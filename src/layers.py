import random
import numpy as np
from utility import *

class Convolution:
    def __init__(self, in_channels, num_filters, ker_size, padding=(0,0), stride=(1, 1)):
        self.kernel = np.random.randn(num_filters, in_channels, ker_size[0], ker_size[1]) * np.sqrt(2 / ker_size[0])
        self.biases = np.zeros(num_filters)
        self.padding = padding
        self.stride = stride

    '''
    for the convolutional layer, we utilize full vectorizaion and im2col/col2m
    in forward/backward passes to reach realistic training times. we pass in output_size instead of padding
    into im2col in order to avoid having to calculate the output size twice, both inside and outside the function. 
    '''

    def forward(self, X):
        pad_h, pad_w = self.padding
        self.X = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
        F, ker_channels, ker_height, ker_width = self.kernel.shape
        B, C, im_height, im_width = self.X.shape
        stride_h, stride_w = self.stride

        output_height = (im_height - ker_height + stride_h) // stride_h
        output_width = (im_width - ker_width + stride_w) // stride_w

        #reshape kernel for matrix multiplication
        self.im_matrix = im2col(self.X, (ker_height, ker_width), self.stride, (output_height, output_width))

        kernel_matrix = self.kernel.reshape(F, -1)

        output = np.matmul(kernel_matrix[None, :, :], self.im_matrix)
        
        return output.reshape(B, F, output_height, output_width) + self.biases.reshape(1, F, 1, 1)
    
    def backward(self, delta_out): #delta_out -> (B, F, output_height, output_width)\
        _, _, ker_height, ker_width = self.kernel.shape
        B, F, dout_height, dout_width = delta_out.shape
        
        self.grad_b = np.sum(delta_out, axis=(0,2,3)) / B

        delta_out = np.reshape(delta_out,( B, F, dout_height * dout_width)) 

        kernel_matrix = self.kernel.reshape(F, -1)  # (F, channels * ker_height * ker_width)

        dX_col = np.matmul(kernel_matrix.T, delta_out)

        im_mat_T = self.im_matrix.transpose(0, 2, 1)      # (B, D, K)
        dK_col  = np.matmul(delta_out, im_mat_T)         # (B, F, K)

        self.grad_w = np.sum(dK_col, axis=0).reshape(self.kernel.shape) / B
        delta_in = col2im(dX_col, self.X, (ker_height, ker_width), self.stride, self.padding)

        return delta_in


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

    '''
    similarly, we utilize im2col and col2im for the forward and backward passes in the pooling layer. once again, we pad the input X
    beforehand, for consistency. due to the use of these functions, the mask used in the naive implementations is not required in the 
    backwards pass. instead we calculate the indices of the max value in each window and use np.addat to recover the layer gradient.
    '''

    def forward(self, X):
        pad_h, pad_w = self.padding
        self.X = np.pad(X, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
        B, channels, x_height, x_width = X.shape
        stride_h, stride_w = self.stride
        output_height = (x_height + 2*pad_h - self.pool_height + stride_h) // stride_h
        output_width = (x_width + 2*pad_w - self.pool_width + stride_w) // stride_w

        self.im_matrix = im2col(self.X, self.pool_size, self.stride, (output_height, output_width), flatten=False)

        if self.type == "max":
            pooled = np.max(self.im_matrix, axis=(-2, -1))
            
        else: 
            pooled = np.mean(self.im_matrix, axis=(-2, -1))

        return pooled 
    

    def backward(self, delta_out):
        B, C, height, width = self.X.shape
        stride_h, stride_w = self.stride
        out_h, out_w = delta_out.shape[2:]
        if self.type == "max":
            flat = self.im_matrix.reshape(B, C, out_h, out_w, -1)
            max_indices = np.argmax(flat, axis=-1)
            #offsets within the window
            window_row_offset = max_indices // self.pool_width
            window_col_offset = max_indices % self.pool_height

            #top left indices for each window
            row_base = np.arange(out_h) * stride_h
            col_base = np.arange(out_w) * stride_w

            row_base = row_base[None, None, :, None]
            col_base = col_base[None, None, None, :]

            row_idx = row_base + window_row_offset
            col_idx = col_base + window_col_offset

            batch_idx = np.arange(B)[:, None, None, None]
            channel_idx = np.arange(C)[None, :, None, None]

            delta_in = np.zeros_like(self.X)
            np.add.at(delta_in, (batch_idx, channel_idx, row_idx, col_idx ), delta_out)

        else:
            grad_cols = np.repeat(delta_out, self.pool_height * self.pool_width, axis=2) / (self.pool_height * self.pool_width)
            grad_cols = grad_cols.reshape(B * C, self.pool_height * self.pool_width, out_h * out_w)     
            
            delta_in = col2im(grad_cols, self.X, (self.pool_size), self.stride, self.padding)

        return delta_in

    
class Linear:
    def __init__(self, in_dim, out_dim):
        self.biases = np.zeros(out_dim,)
        self.weights = np.random.randn(out_dim, in_dim) * np.sqrt(2 / in_dim)

    def forward(self, x):
        self.x = x
        return x @ self.weights.T + self.biases

    def backward(self, delta_out):
        self.grad_w = delta_out.T @ self.x
        self.grad_b = np.sum(delta_out, axis=0)
        return delta_out @ self.weights

class Batch_NormFC:
    def __init__(self, channels, epsilon=1e-8):
        self.gamma = np.ones(channels)
        self.beta = np.ones(channels)
        self.epsilon = epsilon

    def forward(self, X):
        self.X = X

        self.mean = np.mean(X, axis=0)
        self.variance = np.var(X, axis=0)

        self.X_norm = (X - self.mean) / np.sqrt(self.variance + self.epsilon)

        out = self.X_norm * self.gamma + self.beta
        return out

    def backward(self, delta_out): #shape (B, C)
        M, C = delta_out.shape
        self.grad_w = np.sum(delta_out * self.X_norm, axis=0)
        self.grad_b = np.sum(delta_out, axis=0)

        std_inv = 1. / np.sqrt(self.variance + self.epsilon)

        delta_x_norm = delta_out * self.gamma
        delta_v = np.sum(delta_out * (self.X - self.mean) * (-0.5 * self.gamma * ( self.variance + self.epsilon) ** (-3/2)), axis=0)
        delta_m = np.sum(delta_out * (-self.gamma * std_inv), axis=0) + delta_v * (1/M) * np.sum(-2 * (self.X - self.mean), axis=0)

        delta_in = delta_x_norm * std_inv + delta_v * 2 * (self.X - self.mean) / M + delta_m / M
        return delta_in
    
class Batch_NormConv:
    def __init__(self, channels, epsilon=1e-8):
        self.gamma = np.ones(channels)
        self.beta = np.ones(channels)
        self.epsilon = epsilon

    def forward(self, X):
        self.X = X

        self.mean = np.mean(X, axis=(0, 2, 3), keepdims=True)
        self.variance = np.var(X, axis=(0, 2, 3), keepdims=True)

        self.X_norm = (X - self.mean) / np.sqrt(self.variance + self.epsilon)

        gamma = self.gamma.reshape(1, -1, 1, 1)
        beta = self.beta.reshape(1, -1, 1, 1)

        out = self.X_norm * gamma + beta
        return out

    def backward(self, delta_out):
        B, C, H, W = delta_out.shape
        self.grad_w = np.sum(delta_out * self.X_norm, axis=(0, 2, 3))
        self.grad_b = np.sum(delta_out, axis=(0, 2, 3))

        gamma = self.gamma.reshape(1, -1, 1, 1)
        std_inv = 1. / np.sqrt(self.variance + self.epsilon)

        delta_x_norm = delta_out * gamma
        delta_v = np.sum(delta_out * (self.X - self.mean) * (-0.5 * gamma * ( self.variance + self.epsilon) ** (-3/2)), axis=(0, 2, 3), keepdims=True)
        delta_m = np.sum(delta_out * (-gamma * std_inv), axis=(0, 2, 3), keepdims=True) + delta_v * (1/(B * H * W)) * np.sum(-2 * (self.X - self.mean), axis=(0, 2, 3), keepdims=True)

        delta_in = delta_x_norm * std_inv + delta_v * 2 * (self.X - self.mean) / (B * H * W) + delta_m / (B * H * W)
        return delta_in
    

    

class Relu:
    def forward(self, X):
        self.X = X
        return relu(X)
    
    def backward(self, delta_out):
        return delta_out * relu_prime(self.X)


class SoftMaxCrossEntropy:
    def forward(self, logits, labels): #for use in training only
        self.probs = softmax(logits)
        return cross_entropy(self.probs, labels)
    
    def backward(self, labels):
        return (self.probs - labels) / labels.shape[0]


class Flatten:
    def forward(self, X):
        self.dim = X.shape
        batch_size = self.dim[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, delta_out):
        delta_in = delta_out.reshape(self.dim)
        return delta_in
    
    
class Naive_Convolution:
    def __init__(self, in_channels, num_filters, ker_size, padding=(0,0), stride=(1, 1)):
        self.kernel = np.random.randn(num_filters, in_channels, ker_size[0], ker_size[1]) * np.sqrt(2 / ker_size[0])
        self.biases = np.zeros(num_filters)
        self.padding = padding
        self.stride = stride

    '''
    below are the naive implementations using nested for loops. we reuse cross_correlate for the convolution
    needed in the backwards pass.
    '''

    def naive_forward(self, X):
        self.X = X
        return self.multi_out_cross_correlate(X, self.kernel) + self.bias.reshape(1, -1, 1, 1)
    
    def naive_backward(self, delta_out):
        self.grad_w = np.sum(self.multi_out_cross_correlate(self.X, delta_out)) / delta_out.shape[0]
        self.grad_b = np.sum(delta_out, axis=0) / delta_out.shape[0]

        k_rotate = np.flip(self.kernel, (-1, -2))
        delta_in = np.sum(self.multi_out_cross_correlate(delta_out, k_rotate, mode="full"))
        return delta_in
    
    
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


class Naive_Pooling:
    def __init__(self, pool_size, type="max", padding=(0,0), stride=None):
        self.pool_size = pool_size
        self.pool_height, self.pool_width = pool_size
        self.type = type
        self.padding = padding
        if stride is None:
            self.stride = pool_size
        else:
            self.stride = stride

    '''
    below are the naive implementations for the pooling layer. the backwards case for max pooling is handled by the
    creation of a mask during the forward passed, subsequently applied to the outgoing delta using np.addat in order to 
    find the layer gradient.
    '''

    def naive_forward(self, X):
        self.X = X
        self.route = []
        return np.stack([self.multi_in_pool(X[i], i) for i in range(len(X))])
        
    def multi_in_pool(self, X, sample):
            return np.stack([self.pool(X[j], j, sample) for j in range(len(X))])

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
    
    def naive_backward(self, delta_out):
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