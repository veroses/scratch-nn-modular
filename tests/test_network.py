import numpy as np
import pytest
import scratch_nn.network as net_mod
from scratch_nn.network import Network
from scratch_nn.optimization import SGD


class DummyLayer:
    """Forward: X→X*2; Backward: grad→grad/2"""
    def __init__(self):
        self.params = {}
        self.grads = {}
    def forward(self, X):
        return X * 2
    def backward(self, grad):
        return grad / 2


def test_feedforward_runs_through_all_layers():
    net = Network(layers=[DummyLayer(), DummyLayer()], optimizer=SGD)
    X = np.array([[1.0], [2.0]])
    # Each layer multiplies by 2 → after two layers: X * 4
    out = net.feedforward(X)
    assert np.array_equal(out, X * 4)


def test_update_calls_loss_and_optimizer(monkeypatch):
    # Stub out cross_entropy and cross_entropy_delta
    def fake_loss(probs, Y):
    # probs should be (batch_size, num_classes),
    # so here (1,1).  We just check batch-size consistency:
        assert probs.shape[0] == Y.shape[0] == 1
        return 1.23
    def fake_delta(probs, Y):
        return np.array([0.0])
    monkeypatch.setattr(net_mod, 'cross_entropy', fake_loss)
    monkeypatch.setattr(net_mod, 'cross_entropy_delta', fake_delta)

    # Recorder to capture optimizer.update calls
    updates = []
    class RecOpt(SGD):
        def update(self, pg):
            updates.append(pg)

    # A layer that just passes data through
    class PassLayer:
        def __init__(self):
            self.params = {'p': np.zeros(1)}
            self.grads  = {'p': np.ones(1)}
        def forward(self, X): return X
        def backward(self, grad): return grad

    net = Network(layers=[PassLayer()], optimizer=RecOpt)
    X, y = np.array([5.0]), np.array([0])
    loss = net.update([(X, y)])
    assert loss == pytest.approx(1.23)
    # optimizer.update should be called exactly once
    assert len(updates) == 1


def test_train_appends_average_losses():
    net = Network(layers=[DummyLayer()], optimizer=SGD)
    # Override update() to return “batch size” as loss
    net.update = lambda mb: len(mb)

    # Create 4 samples → with mini_batch_size=2 → 2 batches/epoch
    sample = (np.array([0.0]), np.array([0]))
    train_data = [sample] * 4

    net.train(train_data, mini_batch_size=2, epochs=2)
    # Per‐epoch total loss = 2 batches * 2.0 = 4.0 → avg = 4.0/2 = 2.0
    assert net.losses == [2.0, 2.0]


def test_evaluate_counts_correct_predictions():
    # Subclass that treats input as “logits” directly
    class IdNet(Network):
        def __init__(self):
            super().__init__(layers=[DummyLayer()], optimizer=SGD)
        def feedforward(self, X):
            return X

    net = IdNet()
    # Now Y is one-hot of length 2
    data = [
        (np.array([0.1, 0.9]), np.array([0, 1])),
        (np.array([0.8, 0.2]), np.array([1, 0])),
        (np.array([0.3, 0.7]), np.array([0, 1])),
    ]
    acc = net.evaluate(data, batch_size=2)
    # argmax([0.1,0.9])=1 matches label 1, etc.
    assert acc == 3