import torch
import os
import numpy
import scipy


def mnist():
    os.chdir("C:\Users\paulw\Desktop\mlops\dtu_mlops\data")
    print(os.listdir("."))
    # exchange with the corrupted mnist dataset
    train = torch.randn(50000, 784)
    test = torch.randn(10000, 784) 
    return train, test
