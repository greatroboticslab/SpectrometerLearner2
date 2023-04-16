import spectrums

from tsai.all import *

#wandb.init(project = 'wandb_saturn_demo', entity="greatroboticslab")

waveCount = 3694
learningRate = 0.0001
numEpochs = 1
batch_size = 1

hidden_dim = 256
layer_dim = 5

dataSize = 200
threshold = 60

trainData = spectrums.MakeNumpySetClipped("Training Data/", threshold, dataSize)
testData = spectrums.MakeNumpySetClipped("Testing Data/", threshold, dataSize)
validData = spectrums.MakeNumpySetClipped("Validation Data/", threshold, dataSize)

outputCount = len(spectrums.existingEntries)

X, y, splits = combine_split_data([trainData[0], testData[0]], [trainData[1], testData[1]])

#dsid = 'NATOPS' 
#X, y, splits = get_UCR_data(dsid, return_split=False)

print(y.shape)
print(y)

tfms  = [None, [Categorize()]]
dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)

dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)

dls.show_batch(sharey=True)
plt.show()

model = LSTM(dls.vars, dls.c)
learn = Learner(dls, model, metrics=accuracy)
learn.save('stage0')

learn.load('stage0')
learn.lr_find()

learn.fit_one_cycle(100, lr_max=1e-3)
learn.save('stage1')

learn.recorder.plot_metrics()
plt.show()