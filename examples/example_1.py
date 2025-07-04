from scratch_nn import *
from datasets.hasyv2_loader import load_hasy_data

#load dataset
train_data, test_data, index_to_label = load_hasy_data("D:/owenl/personal projects/scratch-nn-refined/data/hasy")

layers = [
    Convolution(1, 8, (3, 3), (1, 1)), BatchNormConv(8), Relu(),
    Pooling((2, 2)),
    Convolution(8, 16, (3, 3), (1, 1)), BatchNormConv(16), Relu(),
    Pooling((2, 2)),
    Flatten(),
    Linear(1024, 369),
    SoftMax()
]

#initialize network
net = Network(layers, Adam, learning_rate=0.001)

if __name__ == "__main__":
    #to verify model and training loop, we overfit a small subsample
    small_train = train_data[:1000]

    net.train(small_train, 32, 30, small_train)