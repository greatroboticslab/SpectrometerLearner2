from multiprocessing import cpu_count
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import spectrums
import time
import wandb

wandb.login()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNNet,self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None
        
    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
        
    def init_hidden(self, x):
        #print(type(x), x.size())
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        #t.cuda()
        return [t.to(device) for t in (h0, c0)]

wandb.init(project = 'wandb_saturn_demo', entity="greatroboticslab")

waveCount = 3694
learningRate = 0.0001
numEpochs = 1000
batch_size = 1

hidden_dim = 256
layer_dim = 5

trainData = spectrums.MakeTorchDataSet("Training Data/")
testData = spectrums.MakeTorchDataSet("Testing Data/")
validData = spectrums.MakeTorchDataSet("Validation Data/")

trainLoader = DataLoader(dataset=trainData, batch_size = batch_size, shuffle=True)
testLoader = DataLoader(dataset=testData, batch_size = batch_size, shuffle=True)
validLoader = DataLoader(dataset=validData, batch_size = batch_size, shuffle=True)

outputCount = len(spectrums.existingEntries)


model = RNNNet(waveCount+1, hidden_dim, layer_dim, outputCount).to(device)

wandb.watch(model)

#print(model)

criterion = nn.CrossEntropyLoss()
#optimizer = Adam(model.parameters(), lr=learningRate)
optimizer = torch.optim.RMSprop(model.parameters(), lr=learningRate)


valAcc = 0
valLoss = 0

vSamples = 0
vCorrect = 0

eMod = 10

lossGraph = []
vGraph = []
vGraphE = []
vAcc = 0

timeArr = [0,0,0,0,0]
timeID = 0

timeStartE = time.time()
timeEndE = 0

learningRate = float(input("Enter learning rate: "))
numEpochs = int(input("How many epochs? "))

for epoch in range(numEpochs):
    for batch_idx, (data, targets) in enumerate(trainLoader):
        
        data = data.to(device=device)
        targets = targets.type(torch.long).to(device)
        
        
        data = data.reshape(1, 3695, data.shape[0])
        #data = data.reshape(data.shape[0], -1)
        #data = data.repeat(1,data.shape[0],1)
        #print(data.shape)
        #data = data.transpose(0, 1).contiguous()
        
        #embed_dim = 10
        #embedLayer = nn.Embedding(int(waveCount+1), embed_dim)
        #data = embedLayer(data)
        
        #print(data)
        #print(targets)
        
        scores = model(data)
        loss = criterion(scores,targets)
        
        #lossGraph.append(float(loss))
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        logs = {
                    'train/train_loss': float(loss),
                    'train/epoch': epoch,
                    'train/accuracy': vAcc,
                    'train/validation_accuracy': valAcc,
                    'train/validation_loss': valLoss
                }
        
        _, predictions = scores.max(1)
        vCorrect += (predictions == targets).sum()
        vSamples += predictions.size(0)
    
    if epoch % eMod == 0:
        timeEndE = time.time()
        timeDif = timeEndE - timeStartE
        timeArr[timeID] = timeDif
        timeStartE = time.time()
        avgTimeDif = 0
        for td in timeArr:
            avgTimeDif += td
        avgTimeDif /= 5
        predictedTime = avgTimeDif*(numEpochs-epoch)
        predictedTime /= 60
        print("Epoch " + str(epoch) + "/" + str(numEpochs) + "\tETA: " + str(predictedTime) + "min.")
        vAcc = float(vCorrect/vSamples)*100
        vCorrect = vSamples = 0
        #ven = (epoch, vAcc)
        
        valLoss = 0
        valCorrect = valSamples = 0
        for batch_idx2, (data2, targets2) in enumerate(validLoader):
            
            data2 = data2.to(device=device)
            targets2 = targets2.type(torch.long).to(device)
            
            data2 = data2.reshape(1, 3695, data.shape2[0])
            #print(data2.shape)
            #data2 = data2.reshape(data2.shape[0], -1)
            #data2 = data2.repeat(1,data2.shape[0],1)
            #print("DATA 2")
            #print(data2.shape)
            
            scores2 = model(data2)
            _2, predictions2 = scores2.max(1)
            
            valCorrect += (predictions2 == targets2).sum()
            valSamples += predictions2.size(0)
            
            valLoss += criterion(scores2,targets2)
        
        
        valLoss = float(valLoss)/valSamples
        
        valAcc = float(valCorrect/valSamples)*100
        
        print("Validation accuracy: " + str(valAcc))
        print("Validation loss: " + str(valLoss))
        
        vGraphE.append(epoch)
    
    timeID += 1
    if timeID > 4:
        timeID = 0
        
    wandb.log(logs)