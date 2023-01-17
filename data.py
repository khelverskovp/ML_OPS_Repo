import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def mnist():
    #download the dataset into train and test sets
    path = "C:/Users/khelv/dtu_mlops/data/corruptmnist"

    Xtrain = np.load(path + "/train_0.npz")["images"]
    Ytrain = np.load(path + "/train_0.npz")["labels"]

    for i in range(1,5):
        Xnext = np.load(path + "/train_{}.npz".format(i))["images"]
        Xtrain = np.concatenate((Xtrain,Xnext),axis=0)
        Ynext = np.load(path + "/train_{}.npz".format(i))["labels"]
        Ytrain = np.concatenate((Ytrain,Ynext),axis=0)

    Xtest = np.load(path + "/test.npz")["images"]
    Ytest = np.load(path + "/test.npz")["labels"]

    #convert to tensors
    Xtrain = torch.from_numpy(Xtrain).float()
    Ytrain = torch.from_numpy(Ytrain).float()
    Xtest = torch.from_numpy(Xtest).float()
    Ytest = torch.from_numpy(Ytest).float()

    train= torch.utils.data.TensorDataset(Xtrain, Ytrain)
    test = torch.utils.data.TensorDataset(Xtest,Ytest)

    return train, test
