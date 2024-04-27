# SpectrometerLearner

## Every python file will use Training Data/ and Testing Data/ folders located in the same directory as the python file.

# vanilla_train.py

vanilla_train.py is a simple vanilla neural network (no convolution, just simple network with 1 hidden layer). To run simply do:
  python vanilla_train.py
and it will use the Training Data/ and Testing Data/ folders.

# train_cnn.py

train_cnn.py is a convolutional neural network with a convolution and pooling layer. To run:
  python train_cnn.py
and it will also use the Training Data and Testing Data folders.

# KNNSpectrometer/main.py

main.py runs a K-Nearest Neighbor algorithm.

# RNNLearner/rnn_learner.py

rnn_learner runs a recurrent neural network.

# SVMLearner/train.py

train.py runs a Support Vector Machines (SVM) algorithm. Previous tests often show this is the most accurate method.

# tsai-main

rnn_learner.py trains a neural network which uses the TSAI library. At the moment it appears to use a LSTM network instead of a transformers network, but can be changed to use transformers.
