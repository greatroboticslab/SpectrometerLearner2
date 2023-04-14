import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import csv

trainDir = "Train Data"
testDir = "Test Data"
peaks = []
subjectList = []

def most_frequent(List):
    return max(set(List), key = List.count)

def GuessSubject(df, filename, k):
    with open(filename, 'r') as sfile:
        
        sfile = csv.reader(sfile, delimiter=",")
        
        subject = ""
        wave = 0.0
        intensity = 0
        
        first = True
        
        for row in sfile:
            if first:
                first = False
            else:
                subject = row[0]
                wave = float(row[1])
                intensity = float(row[2])
                
        validList = []
        
        for index, row in df.iterrows():
            if float(row["wavelength"]) == wave:
                vent = (row["subject"],row["intensity"])
                validList.append(vent)
                
        df2 = pd.DataFrame(np.array(validList), columns=["subject", "intensity"])
        #Get k closest neighbors
        
        neighbors = []
        usedNeighbors = []
        
        for ik in range(k):
            
            lowest = 1000000
            lowestID = -1
            for index, row in df2.iterrows():
                if index not in usedNeighbors:
                    dist = abs(float(row["intensity"]) - intensity)
                    if dist < lowest:
                        lowest = dist
                        lowestID = index
            if lowestID != -1:
                usedNeighbors.append(lowestID)
                neighbors.append(df2.iloc[lowestID]["subject"])
        
        return(most_frequent(neighbors))

spectrumMode = False

for filename in os.listdir(trainDir):
    f = os.path.join(trainDir, filename)
    # checking if it is a file
    
    if os.path.isfile(f):
        extension = os.path.splitext(filename)[1]
        if extension == ".csv":
            fileData = pd.read_csv(f)
            
            if spectrumMode:
                
                if fileData["Sent Wavelength"][0] == 800.0 or True:
                
                    highest = 0.0
                    wave = 0.0
                    
                    #generate subject class string
                    subject = fileData["Subject"][0]
                    
                    subtype = fileData["Subtype"][0]
                    healthy = fileData["Healthy"][0]
                    
                    subjectName = subject + "_" + subtype + "_"
                    
                    
                    
                    if(healthy == "yes"):
                        subjectName += "healthy"
                    else:
                        subjectName += "unhealty"
                    
                    if subjectName not in subjectList:
                        subjectList.append(subjectName)
                    
                    print(subjectName)
                    
                    for index, row in fileData.iterrows():
                        #print(row["Returned Intensity"])
                        if float(row["Returned Intensity"]) > highest:
                            highest = float(row["Returned Intensity"])
                            wave = float(row["Returned Wavelength"])
                            
                    wave = float(fileData["Sent Wavelength"][0])
                    dout = [subjectName, wave, highest]
                    peaks.append(dout)
                    
            else:
                subject = fileData["Subject"][0]
                wave = float(fileData["Sent Wavelength"][0])
                intensity = float(fileData["Intensity"][0])
                
                if subject not in subjectList:
                    subjectList.append(subject)
                
                print(subject)
                dout = [subject, wave, intensity]
                peaks.append(dout)
            

df = pd.DataFrame(np.array(peaks), columns=["subject", "wavelength","intensity"])

df.intensity = df.intensity.astype(float)

df.sort_values('intensity', inplace=True)


#GuessSubject(df, "C:/Programs/KNNSpectrometer/Data/1_hand.csv",3)



kVal = 15


correct = 0
incorrect = 0
total = 0

#test accuracy
for filename in os.listdir(testDir):
    f = os.path.join(testDir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        extension = os.path.splitext(filename)[1]
        if extension == ".csv":
        
            with open(testDir+"/"+filename) as csvFile:
                sfile = csv.reader(csvFile, delimiter=",")
                first = True
            
                actualSubject = ""
            
                total += 1
            
                for row in sfile:
                    if first:
                        first = False
                    else:
                        actualSubject = row[0]
                
                predictedSubject = GuessSubject(df, f, kVal)

                #print(predictedSubject + " = " + actualSubject)

                if predictedSubject == actualSubject:
                    correct += 1
                else:
                    incorrect += 1

accuracy = float(correct/total)
accuracy *= 100

print("Accuracy: " + str(accuracy) + "%. " + str(correct) + "/" + str(total) + " correct.")

colorList = []

#Generate colors
for s in subjectList:
    color = "%06x" % random.randint(0, 0xFFFFFF)
    color = '#' + color
    print(color)
    colorList.append(color)

#colormap

colorMap = []
for index, row in df.iterrows():
    cnt = 0
    for sl in subjectList:
        if row["subject"] == sl:
            colorMap.append(colorList[cnt])
        cnt += 1

fig, ax = plt.subplots()
df.plot('wavelength', 'intensity', kind='scatter', ax=ax, c=colorMap)

for k, v in df.iterrows():
    x = v["wavelength"]
    y = v["intensity"]
    ax.annotate(v["subject"], [x,y])
    
plt.show()
    
print(df)