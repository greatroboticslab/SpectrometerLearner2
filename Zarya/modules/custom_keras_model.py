import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,SimpleRNN, Dense, BatchNormalization, Dropout
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

class CustomKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_shape, num_classes, units=64, dropout_rate=0.5, epochs=10, batch_size=32):
        self.units = units
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        model = Sequential()
        model.add(SimpleRNN(self.units, input_shape=(self.input_shape, 1), return_sequences=True))
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        # You can add more dense layers here
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))

        model.add(Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, X, y):
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

        # Fit the model with callbacks
        model.fit(X.reshape(X.shape[0], X.shape[1], 1), y, 
                  epochs=self.epochs, 
                  batch_size=self.batch_size, 
                  verbose=1,
                  callbacks=[lr_scheduler, early_stopping])
        
        self.model_ = model
        return self

    def score(self, X, y):
        y_pred = self.model_.predict(X.reshape(X.shape[0], X.shape[1], 1))
        y_pred_labels = np.argmax(y_pred, axis=1)
        return accuracy_score(y, y_pred_labels)