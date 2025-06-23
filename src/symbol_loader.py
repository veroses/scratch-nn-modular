import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle

def load_hasy_data(data_dir, test_size=0.2, cache_path="hasy_cache.pkl"):
    if os.path.exists(cache_path):
        print("loading from cache...")
        with open("hasy_cache.pkl", "rb") as f:
            training_data, test_data, index_to_label = pickle.load(f)
        return training_data, test_data, index_to_label
    
    print("loading the slow way...")

    csv_path = os.path.join(data_dir, "hasy-data-labels.csv")
    df = pd.read_csv(csv_path)

    images = []
    labels = []

    for idx, row in df.iterrows():
        img_path = os.path.join(data_dir, row["path"])
        label = row["symbol_id"]

        with Image.open(img_path).convert("L") as img:  # grayscale
            img_arr = np.asarray(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)

        images.append(img_arr)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)   

    unique_labels = np.unique(labels)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    y = np.array([label_to_index[lbl] for lbl in labels])
    y_onehot = np.eye(len(unique_labels))[y]

    x_train, x_test, y_train, y_test = train_test_split(
        images, y_onehot, test_size=test_size, random_state=42
    )

    training_data = [(x, y.reshape(-1, 1)) for x, y in zip(x_train, y_train)]
    test_data = [(x, y.reshape(-1, 1)) for x, y in zip(x_test, y_test)]

    with open("hasy_cache.pkl", "wb") as f:
        pickle.dump((training_data, test_data, index_to_label), f)

    

    return training_data, test_data, index_to_label
