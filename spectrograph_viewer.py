import matplotlib.pyplot as plt
import sys
import csv

def ReadFile(sfile):

    sfile = csv.reader(sfile, delimiter=",")
    
    first = True
    
    waves = []
    intensities = []
    
    for row in sfile:
        if first:
            first = False
        else:
            subject = row[0]
            subtype = row[1]
            healthy = row[2]
            sentWavelength = row[3]
            wavelength = row[4]
            intensity = row[5]
            
            waves.append(float(wavelength))
            intensities.append(float(intensity))

    plt.plot(waves,intensities)
    plt.ylabel("intensity")
    plt.title(subject + " at " + sentWavelength + "nm")
    plt.show()

if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as spfile:
        ReadFile(spfile)
else:
    with open("Sample Data/1_hand.csv", 'r') as spfile:
        ReadFile(spfile)