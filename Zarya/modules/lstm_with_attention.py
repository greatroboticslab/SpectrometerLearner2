import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Permute, Dot, Multiply, Concatenate, Bidirectional
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

class LSTMWithAttention:
    def __init__(self, input_shape, num_classes, units=64, dropout_rate=0.5, epochs=10, batch_size=32, l2_rate = 0.01):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def build_model(self):
        inputs = Input(shape=(self.input_shape, 1))

        lstm_out = LSTM(self.units, return_sequences=True, kernel_regularizer=l2(self.l2_rate))(inputs)

        attention_weights = Dense(1, activation='tanh')(lstm_out)
        attention_weights = Permute((2, 1))(attention_weights)
        attention_weights = tf.keras.layers.Softmax()(attention_weights)
        attention_weights = Permute((2, 1))(attention_weights)

        attention_output = Multiply()([lstm_out, attention_weights])
        attention_output = tf.reduce_sum(attention_output, axis=1)

        dense = Dense(64, activation='relu')(attention_output)
        dense = tf.keras.layers.Dropout(self.dropout_rate)(dense)

        outputs = Dense(self.num_classes, activation='softmax')(dense)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    def fit(self, X, y, X_val=None, y_val=None, plot=False):
        if type(X) is not np.ndarray:
            X = X.to_numpy()
        model = self.build_model()

        # Learning Rate Scheduler callback
        def lr_schedule(epoch):
            lr = 0.001
            if epoch > 50:
                lr *= 0.1
            return lr

        lr_scheduler = LearningRateScheduler(lr_schedule)

        # Early Stopping callback
        early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

        validation_data = None
        if X_val is not None and y_val is not None:
            if type(X_val) is not np.ndarray:
                X_val = X_val.to_numpy()
            validation_data = (X_val.reshape(X_val.shape[0], X_val.shape[1], 1), y_val)
        # Fit the model with callbacks
        history = model.fit(X.reshape(X.shape[0], X.shape[1], 1), y, 
                epochs=self.epochs, 
                batch_size=self.batch_size,
                verbose=1,
                validation_data=validation_data,
                callbacks=[lr_scheduler, early_stopping])
        if plot:
            train_loss = history.history['loss']
            val_loss = history.history['val_loss']
            accuracy = history.history['accuracy']
            val_accuracy = history.history['val_accuracy']

            # Plot the training and validation loss
            plt.plot(train_loss, label='Training Loss')
            plt.plot(val_loss, label='Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

            # Plot the training and validation accuracy
            plt.plot(accuracy, label='Training Accuracy')
            plt.plot(val_accuracy, label='Validation Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()
        
        self.model_ = model
        return self