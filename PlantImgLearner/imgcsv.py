from PIL import Image
from numpy import asarray
import numpy
import os
import csv
import torch
from torch.utils.data import TensorDataset

def MakeDataset(data_dir):
    
    fileFound = True
    id = 1;
    
    tempImages = []
    tempCSV = []
    
    while(fileFound):
        if(os.path.isfile(data_dir+"/"+str(id)+".csv")):
            csvName = data_dir+"/"+str(id)+".csv"
            
            #Get CSV data
            with open(csvName, newline='') as csvFile:
                csvReader = csv.reader(csvFile, delimiter=',')
                first = True
                elements = []
                for row in csvReader:
                    if(first):
                        first = False
                    else:
                        for c in row:
                            if(c == 'yes'):
                                elements.append(1.0)
                            else:
                                elements.append(0.0)
                        tempCSV.append(elements)
            
            
            #Get image data
            imgName = ""
            if(os.path.isfile(data_dir+"/"+str(id)+".jpg")):
                imgName = data_dir+"/"+str(id)+".jpg"
            else:
                imgName = data_dir+"/"+str(id)+".png"
            
            image = Image.open(imgName)
            image = image.resize((64,64))
            
            data = asarray(image)
            data = data.astype('float32')
            data /= 255.0
            
            tempImages.append(data)
            
            
        else:
            fileFound = False
        
        id += 1
        
    inputs = numpy.array(tempImages)
    
    inputs = numpy.transpose(inputs, (0,3,2,1))
    targets = numpy.array(tempCSV)
    
    inputData = torch.Tensor(inputs)
    
    tgtData = torch.Tensor(targets)

    output = TensorDataset(inputData, tgtData)
    
    return output