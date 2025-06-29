from load_data import load_hasy_data, mnist_load_data
from network import LeNet5
import numpy as np

train_data, test_data, id_to_label = load_hasy_data("data/hasy")


net = LeNet5("Adam", learning_rate=0.002, beta1=0.85)
small = train_data[:1000]
net.train(train_data, 32, 50, test_data)


