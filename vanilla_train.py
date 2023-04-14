import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn.functional import normalize
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import csv
import os
import time
import wandb

wandb.login()

from torch.autograd import Variable

class SpectrumDataset(Dataset):
    def __init__(self, labels, waves, train):
        self.labels = labels
        self.waves = waves
        self.train = train
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        labels = self.labels[idx]
        waves = self.waves[idx]
        output = labels, waves
        return output

class DigitNet(nn.Module):

    
    def __init__(self, input_size, hidden1_size, hidden2_size, num_classes):
        super(DigitNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1_size);
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1_size, hidden2_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2_size, num_classes)
        #self.s3 = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out
        
class DigitNet2(nn.Module):

    def __init__(self, input_size, num_classes):
        super(DigitNet2, self).__init__()
        self.fc1 = nn.Linear(input_size, 50);
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, num_classes)
        #self.s3 = nn.Sigmoid()
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
        
class SpectrumNet(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(SpectrumNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size);
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
        
wandb.init(project = 'wandb_saturn_demo', entity="greatroboticslab")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


waveCount = 3694
learningRate = 0.0001
numEpochs = 1
batch_size = 64
hiddenNodes = 50
applyTransform = False

learningRate = float(input("Please enter the learning rate: "))
hiddenNodes = int(input("Please enter the amount of hidden nodes: "))
numEpochs = int(input("How many epochs: "))
#uinput = input("Use transforms? (y/n): ")
uinput = 'f'
if uinput == 'y':
    applyTransform = True

#Load datasets

fileName = "172_hand.csv"

spectrumInputs = []
existingSubjects = []
existingEntries = []

def SameType(subject, subtype, healthy, wavelength, _subject, _subtype, _healthy, _wavelength):
    if(subject == _subject and subtype == _subtype and healthy == _healthy and wavelength == _wavelength):
        return True
    return False
    
#dataDir = "Data/"

def MakeDataSet(dataDir):

    spectrumInputs = []

    for filename in os.listdir(dataDir):
        extension = filename[len(filename)-4:]
        #print(filename[len(filename)-4:])
        if(extension == ".csv"):
        

            with open(dataDir+filename) as csvFile:
                
                plots = csv.reader(csvFile, delimiter=',')
                
                pNum = 0
                waves = []
                subject = ""
                subtype = ""
                healthy = ""
                sent_wavelength = 0.0
                
                for rows in plots:
                    if(pNum == 0):
                        pNum += 1
                    else:
                    
                        if(pNum == 1):
                            pNum += 1
                            subject = rows[0]
                            subtype = rows[1]
                            healthy = rows[2]
                            sent_wavelength = float(rows[3])
                            
                        
                        wavelength = float(rows[4])
                        intensity = (rows[5])
                        
                        waves.append([wavelength,intensity])
                        
                
                entry = [subject,subtype,healthy,sent_wavelength, waves]
                spectrumInputs.append(entry)


    converted = []

    c = -1

    #make datasets
    toConvert = []

    for entry in spectrumInputs:

        nEnt = []
        #first entry is sent wavelength
        nEnt.append(float(entry[3]))
        for wave in entry[4]:
            nEnt.append(float(wave[1]))
            
        toConvert.append(nEnt)
        
    #print(toConvert)

    #just ints
    toConvertTargets = []

    for entry in spectrumInputs:
        exists = False
        for _entry in existingEntries:
            if(SameType(entry[0],entry[1],entry[2],entry[3],_entry[0],_entry[1],_entry[2],_entry[3])):
                exists = True
        #Add new existing entry (an entry of this subject, subtype, health, and wavelength exist)
        if(not exists):
            nEntry = [entry[0], entry[1], entry[2], entry[3]]
            existingEntries.append(nEntry)
            
    #Now create matching labels/targets

    for entry in spectrumInputs:
        for i in range(len(existingEntries)):
            #print(entry[:4])
            #print(existingEntries[i])
            if(SameType(entry[0],entry[1],entry[2],entry[3],existingEntries[i][0],existingEntries[i][1],existingEntries[i][2],existingEntries[i][3])):
                toConvertTargets.append(i)

    #Output dictionary file

    cDictStr = "id,subject,subtype,healthy,wavelength\n"

    did = 1

    for entry in existingEntries:

        cDictStr += str(did) + "," + str(entry[0]) + "," + str(entry[1]) + "," + str(entry[2]) + "," + str(entry[3]) + "\n"
        did += 1
        
    fDict = open("Models/dict.csv", "w")
    fDict.write(cDictStr)
    fDict.close()

    inputData = torch.FloatTensor(toConvert)

    #Apply transforms
    if applyTransform:
        inputData = normalize(inputData)
        for i in range(len(inputData)):
            inputData[i][0] = toConvert[i][0]
        
    #print(inputData)
    
    tgtData = torch.Tensor(toConvertTargets)

    trainData = SpectrumDataset(tgtData, inputData, train=True)
    
    return trainData

trainData = MakeDataSet("Training Data/")
testData = MakeDataSet("Testing Data/")
validData = MakeDataSet("Validation Data/")


trainLoader = DataLoader(dataset=trainData, batch_size = batch_size, shuffle=True)
testLoader = DataLoader(dataset=testData, batch_size = batch_size, shuffle=True)
validLoader = DataLoader(dataset=validData, batch_size = batch_size, shuffle=True)

outputCount = len(existingEntries)

#Initialize network
model = SpectrumNet(waveCount+1, hiddenNodes, outputCount).to(device)

wandb.watch(model)
#print(model)



criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

eMod = 50 #Measure accuracy every eMod epochs

lossGraph = []
vGraph = []
vGraphE = []
vAcc = 0

valAcc = 0
valLoss = 0

vSamples = 0
vCorrect = 0

timeStartE = time.time()
timeEndE = 0

timeArr = [0,0,0,0,0]
timeID = 0

for epoch in range(numEpochs):
    for batch_idx, (targets, data) in enumerate(trainLoader):
        
        #targets, data = trainLoader[i]
        
        #print("THEDATA")
        print(data.shape)
        #print(targets)
        data = data.to(device=device)
        targets = targets.type(torch.long).to(device)
        
        data = data.reshape(data.shape[0], -1)
        #print(data.shape)
        
        scores = model(data)
        loss = criterion(scores,targets)
        
        lossGraph.append(float(loss))
        
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        
        
        #LOG DATA
        logs = {
                    'train/train_loss': float(loss),
                    'train/epoch': epoch,
                    'train/accuracy': vAcc,
                    'train/validation_accuracy': valAcc,
                    'train/validation_loss': valLoss
                }
        
        
        #Correct?
        
        _, predictions = scores.max(1)
        #print(scores)
        #print(predictions)
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
        for batch_idx2, (targets2, data2) in enumerate(validLoader):
            
            data2 = data2.to(device=device)
            targets2 = targets2.type(torch.long).to(device)
            
            data2 = data2.reshape(data2.shape[0], -1)
            
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
        
        

def check_accuracy(loader, model):

    if loader.dataset.train:
        print("Checking training data")
    else:
        print("Checking test data")

    correct = 0
    samples = 0
    model.eval()
    
    #Do not compute gradients (this is the test case)
    with torch.no_grad():
        for entry in loader:
        
            #print("TESTCASE")
            x = entry[1]
            y = entry[0]
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)
            
            scores = model(x)
            
            #print(scores.shape)
            _, predictions = scores.max(1)
            #print(scores)
            #print(predictions)
            correct += (predictions == y).sum()
            samples += predictions.size(0)
            
        print(f'Got {correct} / {samples} with accuracy  {float(correct)/float(samples)*100:.2f}')
        accuracy = float(correct)/float(samples)*100
        
    model.train()
    return accuracy
    
print("Now checking final accuracy...")
    
print("Train Data:")
check_accuracy(trainLoader, model)

print("Test Data:")
check_accuracy(testLoader, model)

torch.save(model.state_dict(), "Models/specreader.cpkt")

#plt.title("Loss over Epochs")
#plt.plot(lossGraph)
#plt.plot(vGraphE, vGraph)
#plt.plot(lossGraph)
#plt.show()

#check_accuracy(testLoader, model)