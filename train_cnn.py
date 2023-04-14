from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import MaxPool1d
from torch.nn import ReLU
from torch.nn import LogSoftmax
from torch import flatten

import spectrums

from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch import nn
import numpy as np
import torch
import time

class SpectrumNet(Module):
    def __init__(self, numChannels, classes):
        super(SpectrumNet, self).__init__()
        
        #CONV, RELU, POOL
        self.conv1 = Conv1d(in_channels=numChannels, out_channels=20, kernel_size=(5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool1d(kernel_size=(2), stride=(2))
        
        self.conv2 = Conv1d(in_channels=20, out_channels=50, kernel_size=(5))
        self.relu2 = ReLU()
        self.maxpool2 = MaxPool1d(kernel_size=(2), stride=(2))
        
        self.fc1 = Linear(in_features=80, out_features=30)
        self.relu3 = ReLU()
        
        self.fc2 = Linear(in_features=30, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)
        
    def forward(self, x):
    
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        
        #x = self.conv2(x)
        #x = self.relu2(x)
        #x = self.maxpool2(x)
        
        #x = flatten(x,1)
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        output = self.logSoftmax(x)
        
        return output
        

class MyNet(nn.Module):
    def __init__(self, input_size, num_output_layers):
        super(MyNet, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * input_size, num_output_layers)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


waveCount = 3694
learningRate = 0.0001
numEpochs = 1
batch_size = 1

trainData = spectrums.MakeTorchDataSet("Training Data/")
testData = spectrums.MakeTorchDataSet("Testing Data/")
validData = spectrums.MakeTorchDataSet("Validation Data/")

trainLoader = DataLoader(dataset=trainData, batch_size = batch_size, shuffle=True)
testLoader = DataLoader(dataset=testData, batch_size = batch_size, shuffle=True)
validLoader = DataLoader(dataset=validData, batch_size = batch_size, shuffle=True)

outputCount = len(spectrums.existingEntries)

model = MyNet(waveCount+1, outputCount).to(device)

print(model)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learningRate)


for epoch in range(numEpochs):
    for batch_idx, (data, targets) in enumerate(trainLoader):
        
        data = data.to(device=device)
        targets = targets.type(torch.long).to(device)
        
        print(data.shape)
        #data = data.reshape(data.shape[0], -1)
        data = data.transpose(0, 1).contiguous()
        print(data)
        print(targets)
        
        scores = model(data)
        loss = criterion(scores,targets)
        
        lossGraph.append(float(loss))
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        _, predictions = scores.max(1)
        vCorrect += (predictions == targets).sum()
        vSamples += predictions.size(0)