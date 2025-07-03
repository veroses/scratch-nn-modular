import numpy as np
import pytest

from scratch_nn import Convolution, Pooling, Linear, Flatten, BatchNormConv, BatchNormFC

def almost_equal(a, b, tol=1e-6):
    return np.allclose(a, b, atol=tol)


# ─── Linear ────────────────────────────────────────────────────────────────────

def test_linear_forward_backward_and_grads():
    B, in_dim, out_dim = 4, 3, 2
    X = np.random.randn(B, in_dim)
    delta = np.random.randn(B, out_dim)

    layer = Linear(in_dim, out_dim)
    # check shapes
    assert layer.weights.shape == (out_dim, in_dim)
    assert layer.biases.shape  == (out_dim,)

    # forward
    out = layer.forward(X)
    assert out.shape == (B, out_dim)
    # manual forward
    expected_out = X @ layer.weights.T + layer.biases
    assert almost_equal(out, expected_out)

    # backward
    back = layer.backward(delta)
    assert back.shape == X.shape
    # grads
    # w-grad = delta.T @ X
    expected_w_grad = delta.T @ X
    expected_b_grad = np.sum(delta, axis=0)
    assert almost_equal(layer.grads["w"], expected_w_grad)
    assert almost_equal(layer.grads["b"], expected_b_grad)


# ─── Flatten ───────────────────────────────────────────────────────────────────

def test_flatten_forward_backward_inverts_shape():
    B, C, H, W = 2, 3, 4, 5
    X = np.random.randn(B, C, H, W)
    layer = Flatten()

    out = layer.forward(X)
    assert out.shape == (B, C*H*W)

    back = layer.backward(out)
    assert back.shape == X.shape
    assert almost_equal(back, X)


# ─── Pooling ───────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ptype", ["max", "avg"])
def test_pooling_forward_backward_identity(ptype):
    # Use a 1×1 channel, 4×4 image so we can manual-check
    X = np.arange(16).reshape(1,1,4,4).astype(float)
    layer = Pooling(pool_size=(2,2), type=ptype, padding=(0,0), stride=(2,2))

    out = layer.forward(X)
    # out shape = (1,1,2,2)
    assert out.shape == (1,1,2,2)

    # For max: manually compute; for avg: should be mean of each 2×2 block
    if ptype == "max":
        expected = np.array([[[[ 5,  7],
                               [13, 15]]]])
    else:
        expected = np.array([[[[  (0+1+4+5)/4,  (2+3+6+7)/4 ],
                                [ (8+9+12+13)/4, (10+11+14+15)/4 ]]]])
    assert almost_equal(out, expected)

    # Now check backward 
    # Use delta_out = ones; sum of backward should equal sum(delta_out) for avg,
    # and for max should distribute only to max positions.
    delta_out = np.ones_like(out)
    delta_in = layer.backward(delta_out)
    assert delta_in.shape == X.shape

    if ptype == "avg":
        # each backward pixel in a block gets delta_out/4 = 0.25
        assert almost_equal(delta_in, 0.25 * np.ones_like(X))
    else:
        # for max: only the max indices receive gradient = 1
        mask = np.zeros_like(X)
        # positions [0,1], [1,1] in each 2×2 block
        mask[0,0,1,1] = 1  # covers indices for 2×2 blocks within 4×4?
        # actually two blocks: top-left block: max at (0,1); top-right at (0,3);
        # bottom-left at (2,1); bottom-right at (2,3)
        mask = np.zeros_like(X)
        mask[0,0,0*2+1,0*2+1] = 1
        mask[0,0,0*2+1,1*2+1] = 1
        mask[0,0,1*2+1,0*2+1] = 1
        mask[0,0,1*2+1,1*2+1] = 1
        assert almost_equal(delta_in, mask)


# ─── Convolution ───────────────────────────────────────────────────────────────

def test_convolution_identity_1x1_kernel_forward_backward():
    # identity conv: in_channels = num_filters = 2, kernel = 1×1, stride=1, padding=0
    B, C, H, W = 3, 2, 5, 5
    X = np.random.randn(B, C, H, W)
    layer = Convolution(in_channels=C, num_filters=C, ker_size=(1,1),
                        padding=(0,0), stride=(1,1))

    # override kernel to identity
    kernel = np.zeros_like(layer.kernel)
    for i in range(C):
        kernel[i,i,0,0] = 1.0
    layer.kernel = kernel
    layer.params["k"] = layer.kernel
    layer.biases = np.zeros(C)
    layer.params["b"] = layer.biases

    # forward should be exact identity
    out = layer.forward(X)
    assert out.shape == X.shape
    assert almost_equal(out, X)

    # backward with random delta
    delta = np.random.randn(*out.shape)
    back = layer.backward(delta)
    assert back.shape == X.shape
    # gradient of kernel k[i,i,0,0] = average over batch of sum of delta[:,i,:,:]
    grad_k = layer.grads["k"]
    B = X.shape[0]
    d = delta.reshape(B, C, -1)
    x = X.reshape(B, C, -1)
    expected = np.array([
    np.sum(d[:, i, :] * x[:, i, :]) / B
    for i in range(C)
])
    for i in range(C):
        assert pytest.approx(grad_k[i,i,0,0], rel=1e-6) == expected[i]


# ─── BatchNormFC ───────────────────────────────────────────────────────────────

def test_batchnormfc_forward_stats():
    B, C = 8, 5
    X = np.random.randn(B, C)
    bn = BatchNormFC(channels=C, epsilon=1e-8)
    out = bn.forward(X)

    # shape
    assert out.shape == X.shape

    # with gamma=1, beta=1, output mean ≈ 1, var ≈ 1
    mean = out.mean(axis=0)
    var  = out.var(axis=0)
    assert almost_equal(mean, np.ones(C))
    assert almost_equal(var,  np.ones(C))


def test_batchnormfc_backward_grads_and_shape():
    B, C = 6, 4
    X = np.random.randn(B, C)
    bn = BatchNormFC(channels=C)
    out = bn.forward(X)

    delta = np.ones_like(out)
    delta_in = bn.backward(delta)

    # backward shape matches input
    assert delta_in.shape == X.shape

    # grads: 
    #   g = sum(delta * X_norm, axis=0)
    #   b = sum(delta, axis=0)
    expected_g = np.sum(bn.X_norm, axis=0)
    expected_b = np.sum(delta, axis=0)
    assert almost_equal(bn.grads["g"], expected_g)
    assert almost_equal(bn.grads["b"], expected_b)


# ─── BatchNormConv ────────────────────────────────────────────────────────────

def test_batchnormconv_forward_stats():
    B, C, H, W = 3, 4, 5, 6
    X = np.random.randn(B, C, H, W)
    bn = BatchNormConv(channels=C, epsilon=1e-8)
    out = bn.forward(X)

    # shape
    assert out.shape == X.shape

    # check per-channel mean/var over (N, H, W)
    mean = out.mean(axis=(0, 2, 3))
    var  = out.var(axis=(0, 2, 3))
    assert almost_equal(mean, np.ones(C))
    assert almost_equal(var,  np.ones(C))


def test_batchnormconv_backward_grads_and_shape():
    B, C, H, W = 2, 3, 4, 4
    X = np.random.randn(B, C, H, W)
    bn = BatchNormConv(channels=C)
    out = bn.forward(X)

    delta = np.ones_like(out)
    delta_in = bn.backward(delta)

    # backward shape matches input
    assert delta_in.shape == X.shape

    # grads:
    #   g = sum(delta * X_norm, axis=(0,2,3), keepdims=True)
    #   b = sum(delta, axis=(0,2,3), keepdims=True)
    expected_g = np.sum(bn.X_norm * delta, axis=(0, 2, 3), keepdims=True)
    expected_b = np.sum(delta, axis=(0, 2, 3), keepdims=True)
    assert almost_equal(bn.grads["g"], expected_g)
    assert almost_equal(bn.grads["b"], expected_b)