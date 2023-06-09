import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import spectrums

train_x, train_y = spectrums.MakeNumpySetClipped("Training Data/", 3000, 300)

n_classes = len(np.unique(train_y))

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x,x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs
    
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

input_shape = train_x.shape[1:]

model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_transformer_blocks=4, mlp_units=[128], mlp_dropout=0.4, dropout=0.25)

model.compile(loss="sparse_categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-4), metrics=["sparse_categorical_accuracy"])

model.summary()

callbacks = [keras.callbacks.EarlyStopping(patience=3000, restore_best_weights=True)]

model.fit(train_x, train_y, validation_split=0.2,epochs=3000,batch_size=64,callbacks=callbacks)

model.evaluate(train_x, train_y, verbose=1)