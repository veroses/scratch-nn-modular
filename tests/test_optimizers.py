import numpy as np
import pytest
from scratch_nn import SGD, Adam

def test_sgd_momentum_accumulates():
    sgd = SGD(learning_rate=0.1, momentum=0.5)
    params = {'w': np.array([1.0])}
    grads  = {'w': np.array([1.0])}

    # First update: velocity = -0.1 * grad = -0.1
    sgd.update(params, grads)
    assert pytest.approx(params['w'][0], rel=1e-6) == 1.0 - 0.1
    prev = params['w'][0]

    # Second update: velocity = 0.5*(-0.1) - 0.1*1 = -0.05 - 0.1 = -0.15
    sgd.update(params, grads)
    assert pytest.approx(params['w'][0], rel=1e-6) == prev - 0.15

def test_sgd_update_no_momentum():
    sgd = SGD(learning_rate=0.1, momentum=0.0)
    params = {"w": np.array([1.0, 2.0])}
    grads  = {"w": np.array([0.5, 1.0])}
    sgd.update(params, grads)
    # velocity starts at 0 => param -= lr * grad
    expected = np.array([1.0, 2.0]) - 0.1 * np.array([0.5, 1.0])
    assert np.allclose(params["w"], expected)


def test_sgd_update_with_momentum():
    sgd = SGD(learning_rate=0.1, momentum=0.9)
    params = {"w": np.array([1.0])}
    grads  = {"w": np.array([0.5])}

    # first step
    sgd.update(params, grads)
    # vel = -0.1 * 0.5 = -0.05, param = 1.0 - 0.05
    assert np.allclose(params["w"], 0.95)

    # second step
    sgd.update(params, grads)
    # vel = 0.9*(-0.05) - 0.1*0.5 = -0.045 - 0.05 = -0.095
    # param = 0.95 - 0.095 = 0.855
    assert np.allclose(params["w"], 0.855)


def test_adam_update_first_step():
    adam = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params = {"w": np.array([1.0])}
    grads  = {"w": np.array([0.5])}

    adam.update(params, grads)
    # m = 0.9*0 + 0.1*0.5 = 0.05
    # v = 0.999*0 + 0.001*(0.5**2) = 0.00025
    # m_hat = 0.05 / (1 - 0.9) = 0.5
    # v_hat = 0.00025 / (1 - 0.999) = 0.25
    # update = lr * m_hat / (sqrt(v_hat) + eps)
    update = 0.1 * 0.5 / (np.sqrt(0.25) + 1e-8)
    assert np.allclose(params["w"], 1.0 - update)

def test_sgd_zero_gradient_no_change():
    sgd = SGD(learning_rate=0.1, momentum=0.0)
    params = {'w': np.array([1.0])}
    grads  = {'w': np.array([0.0])}

    sgd.update(params, grads)
    # zero gradient should leave weights unchanged
    assert np.allclose(params['w'], np.array([1.0]))

def test_adam_zero_gradient_no_change():
    adam = Adam(learning_rate=0.1)
    params = {'w': np.array([1.0])}
    grads  = {'w': np.array([0.0])}

    adam.update(params, grads)
    # zero gradient → no parameter update
    assert np.allclose(params['w'], np.array([1.0]))
