import torch
import torch.nn as nn
import torch.nn.functional as F
import imgcsv
from torch.utils.data import TensorDataset, DataLoader

from torchvision import datasets, transforms, models

batch_size = 3


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CNNNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(16384,1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,2)
        )
        
    def forward(self, x):
        return self.network(x)

trainData = imgcsv.MakeDataset("Training Data")

trainLoader = DataLoader(dataset=trainData, batch_size = batch_size, shuffle=True)

print(trainData.shape)

model = CNNNet().to(device)

learningRate = 0.001
numEpochs = 100


criterion = nn.CrossEntropyLoss()

loss_fn = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(numEpochs):
    for batch_idx, (data, targets) in enumerate(trainLoader):
        
        data = data.to(device=device)
        targets = targets.type(torch.long).to(device)
        targets = targets.to(torch.float32)
        
        scores = model(data)
        #loss = criterion(scores,targets)
        loss = loss_fn(scores,targets)
        
        print("Loss: " + str(float(loss)))
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        
