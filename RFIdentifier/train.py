import spectrums
import matplotlib.pyplot as plt
import pickle
from sklearn import tree

trainInputs, trainTargets = spectrums.MakeDataSet("Training Data/")

clf = tree.DecisionTreeClassifier()
clf = clf.fit(trainInputs, trainTargets)

print(spectrums.existingEntries)
print("Size: " + str(len(spectrums.existingEntries)))

testInputs, testTargets = spectrums.MakeDataSet("Testing Data/")

spectrums.SaveDictionary()



predictions = clf.predict(trainInputs)

correct = 0
total = len(predictions)
for x in range(len(predictions)):
    if predictions[x] == trainTargets[x]:
        correct += 1

acc = float(correct/total)*100

print("Train Accuracy: " + str(acc))



predictions = clf.predict(testInputs)

correct = 0
total = len(predictions)
for x in range(len(predictions)):
    if predictions[x] == testTargets[x]:
        correct += 1

acc = float(correct/total)*100

print("Test Accuracy: " + str(acc))

with open("Models/tree.pickle", "wb") as tFile:
    pickle.dump(clf, tFile)

#saved = pickle.dumps(clf)

plt.figure()
tree.plot_tree(clf)
plt.show()