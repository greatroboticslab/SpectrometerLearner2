import spectrums
import matplotlib.pyplot as plt
import pickle
from sklearn import svm
from sklearn.model_selection import cross_val_score

trainInputs, trainTargets = spectrums.MakeDataSet("Training Data/")

testInputs, testTargets = spectrums.MakeDataSet("Testing Data/")

spectrums.SaveDictionary()

#clf = svm.SVC(kernel='linear').fit(trainInputs,trainTargets)

clf = svm.SVC(kernel='linear', C=1).fit(trainInputs,trainTargets)
scores = cross_val_score(clf, trainInputs, trainTargets, cv=2)

print(scores)

correct = 0
total = len(trainTargets)



predictions = clf.predict(trainInputs)

for x in range(len(predictions)):
    if predictions[x] == trainTargets[x]:
        correct += 1


acc = float(correct/total)*100

print("Train Accuracy: " + str(acc))


with open("Models/svm.pickle", "wb") as tFile:
    pickle.dump(clf, tFile)


total = len(testTargets)

correct = 0

predictions = clf.predict(testInputs)

for x in range(len(predictions)):
    if predictions[x] == testTargets[x]:
        correct += 1

acc = float(correct/total)*100

print("Test Accuracy: " + str(acc))
with open("Models/svm.pickle", "wb") as tFile:
    pickle.dump(clf, tFile)