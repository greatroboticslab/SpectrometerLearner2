import os
import csv

spectrumInputs = []
existingSubjects = []
existingEntries = []

def SameType(subject, subtype, healthy, wavelength, _subject, _subtype, _healthy, _wavelength):
    if(subject == _subject and subtype == _subtype and healthy == _healthy and wavelength == _wavelength):
        return True
    return False

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
                
    return (toConvert, toConvertTargets)

def SaveDictionary():
    #Output dictionary file

    cDictStr = "id,subject,subtype,healthy,wavelength\n"

    did = 1

    for entry in existingEntries:

        cDictStr += str(did) + "," + str(entry[0]) + "," + str(entry[1]) + "," + str(entry[2]) + "," + str(entry[3]) + "\n"
        did += 1
        
    with open("Models/dict.csv", "w") as fDict:
        fDict.write(cDictStr)
        fDict.close()
        
def LoadDictionary():
    subjectList = []

    with open("settings/dict.csv") as csvFile:
        
        plots = csv.reader(csvFile, delimiter=',')
        
        first = True
        
        for row in plots:
        
            if first:
                first = False
            else:
                entry = (row[0], row[1], row[2], row[3], row[4])
                subjectList.append(entry)
    return subjectList