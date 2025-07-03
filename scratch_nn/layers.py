import numpy as np
from .utils import *
from abc import ABC, abstractmethod
from .optimization import *

class Layer(ABC):
    def __init__(self):
        self.params = {}
        self.grads = {}

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, X):
        pass

    def set_mode(self, train=True):
        pass


'''
    for the convolutional layer, we utilize full vectorizaion and im2col/col2m
    in forward/backward passes to reach realistic training times. we pass in output_size instead of padding
    into im2col in order to avoid having to calculate the output size twice, both inside and outside the function. 
    '''

class Convolution(Layer):
    def __init__(self, in_channels, num_filters, ker_size, padding=(0,0), stride=(1, 1)):
        self.kernel = np.random.randn(num_filters, in_channels, ker_size[0], ker_size[1]) * np.sqrt(2 / ker_size[0])
        self.biases = np.zeros(num_filters)
        self.padding = padding
        self.stride = stride

        self.params = {"k": self.kernel, "b": self.biases}
        self.grads = {"k": np.zeros_like(self.kernel), "b": np.zeros_like(self.biases)}

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
        
        self.grads["b"] = np.sum(delta_out, axis=(0,2,3)) / B

        delta_out = np.reshape(delta_out,( B, F, dout_height * dout_width)) 

        kernel_matrix = self.kernel.reshape(F, -1)  # (F, channels * ker_height * ker_width)

        dX_col = np.matmul(kernel_matrix.T, delta_out)

        im_mat_T = self.im_matrix.transpose(0, 2, 1)      # (B, D, K)
        dK_col  = np.matmul(delta_out, im_mat_T)         # (B, F, K)

        self.grads["k"] = np.sum(dK_col, axis=0).reshape(self.kernel.shape) / B
        delta_in = col2im(dX_col, self.X, (ker_height, ker_width), self.stride, self.padding)

        return delta_in


'''
    similarly, we utilize im2col and col2im for the forward and backward passes in the pooling layer. once again, we pad the input X
    beforehand, for consistency. due to the use of these functions, the mask used in the naive implementations is not required in the 
    backwards pass. instead we calculate the indices of the max value in each window and use np.addat to recover the layer gradient.
    '''

class Pooling(Layer):
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

    
class Linear(Layer):
    def __init__(self, in_dim, out_dim):
        self.biases = np.zeros(out_dim,)
        self.weights = np.random.randn(out_dim, in_dim) * np.sqrt(2 / in_dim)

        self.params = {"w": self.weights, "b": self.biases}
        self.grads = {"w": np.zeros_like(self.weights), "b": np.zeros_like(self.weights)}

    def forward(self, x):
        self.x = x
        return x @ self.weights.T + self.biases

    def backward(self, delta_out):
        self.grads["w"] = delta_out.T @ self.x
        self.grads["b"] = np.sum(delta_out, axis=0)
        return delta_out @ self.weights


class Flatten(Layer):
    def forward(self, X):
        self.dim = X.shape
        batch_size = self.dim[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, delta_out):
        delta_in = delta_out.reshape(self.dim)
        return delta_in
    