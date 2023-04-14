import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import spectrums
import wandb
from wandb.keras import WandbMetricsLogger

lr = 0.0001
epc = 10
lr = float(input("Please enter the learning rate: "))
epc = int(input("Please enter epoch count: "))

wandb.init(config={"bs":12})

train_x, train_y = spectrums.MakeNumpySet("Training Data/")
train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)
train_y = train_y.reshape(train_y.shape[0], 1)
train_y = tf.keras.utils.to_categorical(train_y)


test_x, test_y = spectrums.MakeNumpySet("Testing Data/")
test_x = test_x.reshape(test_x.shape[0], test_x.shape[1],1)
test_y = test_y.reshape(test_y.shape[0], 1)
test_y = tf.keras.utils.to_categorical(test_y)

num_classes = len(spectrums.existingEntries)

print(train_x.shape)
print(train_y.shape)


model = models.Sequential()
model.add(layers.Conv1D(256, 2, activation='relu', input_shape=(train_x.shape[1:]), strides=1))
model.add(layers.Conv1D(256, 3, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling1D(pool_size=2, strides=1, padding='valid'))
model.add(layers.Flatten())
model.add(layers.Dense(300, activation='relu'))

#model.add(layers.Conv1D(64, 2, activation='relu', input_shape=(train_x.shape[1:])))
#model.add(layers.Dense(90, activation='relu'))
#model.add(layers.MaxPooling1D())
#model.add(layers.Flatten())


#model.add(layers.Dense(4000, input_shape=(train_x.shape[1],), activation='relu'))
#model.add(layers.Dense(256,activation='relu'))


#output layers

model.add(layers.Dense(num_classes, activation="softmax"))

print(model.summary())


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
model.fit(x=train_x, y=train_y, batch_size=64, epochs=epc, validation_data=(test_x,test_y),callbacks=[WandbMetricsLogger()])

#score, acc = model.evaluate(test_x, test_y, batch_size=3)

#print("Score: ", score)
#print("Acc: ", acc)