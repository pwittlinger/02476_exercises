import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

if __name__ == "__main__":
    path = "./data/raw/corruptmnist/"
    print(os.listdir(path))
    content = np.load(path + "train_0.npz", allow_pickle=True)

    print(content.f.images.shape)
    for i in range(10):
        plt.imshow(content.f.images[i])
        plt.show()

    # data = torch.tensor(np.concatenate([c['images'] for c in content])).reshape(-1, 1, 28, 28)
    # targets = torch.tensor(np.concatenate([c['labels'] for c in content]))
    # print(data)
    # plt.imshow(data[0])


def load_data(path: str = "./data/raw/corruptmnist/"):
    train_c = []
    train_l = []
    test_c = []
    test_l = []
    files = os.listdir(path)

    for f in files:
        content = []
        if f.__contains__("train"):
            content.append(np.load(path + f, allow_pickle=True))
            data = torch.tensor(np.concatenate([c["images"] for c in content])).reshape(
                -1, 1, 28, 28
            )
            targets = torch.tensor(np.concatenate([c["labels"] for c in content]))
            plt.imshow(data)
        else:
            content.append(np.load(path + f, allow_pickle=True))
            data = torch.tensor(np.concatenate([c["images"] for c in content])).reshape(
                -1, 1, 28, 28
            )
            targets = torch.tensor(np.concatenate([c["labels"] for c in content]))
            plt.imshow(data)

    return data, targets
