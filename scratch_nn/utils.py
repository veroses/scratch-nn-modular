import numpy as np


def get_indices(X_shape, k_size, stride, output_size):
    B, channels, im_height, im_width = X_shape
    k_height, k_width = k_size
    stride_h, stride_w = stride
    output_height, output_width = output_size

    batch_idx = np.arange(B).reshape(B, 1, 1, 1, 1, 1)
    channel_idx = np.arange(channels).reshape(1, channels, 1, 1, 1, 1)

    x1_idx = np.arange(output_width) * stride_w
    y1_idx = np.arange(output_height) * stride_h

    x_offsets, y_offsets = np.meshgrid(np.arange(k_width), np.arange(k_height), indexing="ij")

    xf_idx = np.add.outer(x1_idx, x_offsets)
    yf_idx = np.add.outer(y1_idx, y_offsets)

    xf_idx = xf_idx.reshape(1, 1, 1, output_width, k_height, k_width)
    yf_idx = yf_idx.reshape(1, 1, output_height, 1, k_height, k_width)

    return batch_idx, channel_idx, xf_idx, yf_idx


def im2col(X, size, stride, output_size, flatten="True"): #X is padded
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
    

def col2im(dX_col, X, size, stride, padding): #X is padded
    pad_h, pad_w = padding
    k_height, k_width = size
    B, channels, im_height, im_width = X.shape
    stride_h, stride_w = stride

    output_height = (im_height - k_height + stride_h) // stride_h
    output_width = (im_width - k_width + stride_w) // stride_w

    im = np.zeros_like(X)

    batch_idx, channel_idx, xf_idx, yf_idx = get_indices(X.shape, size, stride, (output_height, output_width))

    b_idx, c_idx, x_idx, y_idx = np.broadcast_arrays(batch_idx, channel_idx, xf_idx, yf_idx)
    b_idx = b_idx.ravel()
    c_idx = c_idx.ravel()
    x_idx = x_idx.ravel()
    y_idx = y_idx.ravel()
    vals = dX_col.reshape(-1)
    
    np.add.at(im, (b_idx, c_idx, x_idx, y_idx), vals)

    if pad_h == 0 and pad_w == 0:
        return im
    else:
        return im[:, :, pad_h:im.shape[2]-pad_h, pad_w:im.shape[3]-pad_w]