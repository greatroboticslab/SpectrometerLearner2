import os
import spectrums
import matplotlib.pyplot as plt
import pickle
from sklearn import tree

subjectList = spectrums.LoadDictionary()

with open("settings/tree.pickle", "rb") as tFile:
    clf = pickle.load(tFile)

os.system('read_raw.bat')

inputData, _tgt = spectrums.MakeDataSet("Data/")

predictions = clf.predict(inputData)

print(subjectList[predictions[0]])