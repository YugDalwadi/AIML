import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

import torchvision.datasets as datasets
mnist_trainset = datasets.MNIST(root=r'/home/xyphoes/Desktop/Projects/AIML - 2/Learning/Neural Networks/Projects/data', train=True, download=False, transform=None)
mnist_testset = datasets.MNIST(root=r'/home/xyphoes/Desktop/Projects/AIML - 2/Learning/Neural Networks/Projects/data', train=False, download=False, transform=None)

(xTrain, yTrain) = mnist_trainset.data.to(torch.float32), mnist_trainset.targets.to(torch.float32)
(xTest, yTest) = mnist_testset.data.to(torch.float32), mnist_testset.targets.to(torch.float32)
xTrain = xTrain / 255.0
xTest = xTest / 255.0
xTrain = xTrain.unsqueeze(1).to(device)  # shape: (N, 1, 28, 28)
xTest = xTest.unsqueeze(1).to(device)
yTest = yTest.long().to(device)
yTrain = yTrain.long().to(device)
xTrain.shape[0]

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fl1 = nn.Linear(100,256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fl2 = nn.Linear(256,512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fl3 = nn.Linear(512,1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.out = nn.Linear(1024,28*28)
    
    def forward(self, X):
        out = F.relu(self.fl1(X))
        out = self.bn1(out)
        out = F.relu(self.fl2(out))
        out = self.bn2(out)
        out = F.relu(self.fl3(out))
        out = self.bn3(out)
        out = self.out(out)
        out = torch.reshape(out, (X.shape[0],1,28,28))
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fl1 = nn.Linear(28*28,512)
        self.fl2 = nn.Linear(512,256)
        self.out = nn.Linear(256,1)
    
    def forward(self,X):
        out = self.flatten(X)
        out = F.leaky_relu(self.fl1(out), negative_slope=0.2)
        out = F.leaky_relu(self.fl2(out), negative_slope=0.2)
        out = self.out(out)
        # out = nn.Sigmoid(self.out(out))
        return out
    
from torch.utils.data import TensorDataset, DataLoader
train_ds = TensorDataset(xTrain, yTrain)
batch_size = 64
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

genModel = Generator().to(device)
discModel = Discriminator().to(device)
genOptim = torch.optim.Adam(genModel.parameters())
discOptim = torch.optim.Adam(discModel.parameters())
genLoss = nn.BCEWithLogitsLoss()
discLoss = nn.BCEWithLogitsLoss()
n_epoch = 10

genModel.train()
discModel.train()
for epoch in range(n_epoch):
    for xb,yb in train_loader:
        noise = torch.randn(xb.size(0),100).to(device)
        realLabels = torch.ones(xb.size(0),1,device=device)
        fakeLabels = torch.zeros(xb.size(0),1,device=device)
        fakeImages = genModel(noise)
        discPreds_gen = discModel(fakeImages.detach())
        discPreds_real = discModel(xb)
        
        discOptim.zero_grad()
        d_L = discLoss(discPreds_gen, fakeLabels) + discLoss(discPreds_real,realLabels) # to be ascended
        d_L.backward()
        discOptim.step()

        genOptim.zero_grad()
        fakeImages = genModel(noise)
        discPreds_gen = discModel(fakeImages)
        g_L = genLoss(discPreds_gen, realLabels)
        g_L.backward()
        genOptim.step()

    print(f"Gen Loss: {g_L} Disc Loss: {d_L}")
        
genModel.eval()
discModel.eval()

noise = torch.randn(xb.size(0),100).to(device)
img = genModel(noise)
plt.imshow(img)