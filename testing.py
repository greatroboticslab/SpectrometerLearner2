import spectrums

train_x, train_y = spectrums.MakeNumpySetClipped("Training Data/", 30, 200)

print(train_x.shape)