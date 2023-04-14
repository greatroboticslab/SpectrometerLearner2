import os
import shutil
import csv

dir = "Testing Data"

modul = int(input("Save every xth file (excluding green wavelengths): "))

for filename in os.listdir(dir):
    f = os.path.join(dir, filename)
    # checking if it is a file
    if os.path.isfile(f):
        extension = os.path.splitext(filename)[1]
        if extension == ".csv":
            #get number
            underPos = 0
            idx = 0
            found = False
            for ch in filename:
                if ch == '_' and (not found):
                    underPos = idx
                    found = True
                idx += 1
            if int(filename[:underPos]) % modul == 1:
                with open(f, 'r') as asfile:
                    
                    sfile = csv.reader(asfile, delimiter=",")
                    
                    first = True
                    
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
                            
                            if float(sentWavelength) != 530.0:
                                
                                asfile.close()
                                shutil.move(f, "Validation Data/"+filename)
                                print(filename + " moved.")
                            break
                    
                    