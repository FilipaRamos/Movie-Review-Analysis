import numpy as np
import matplotlib.pyplot as plt

import keras

### Callback Class Definition ###
# Create callback class to get the accuracy and loss
class AccuracyLossHistory(keras.callbacks.Callback):
    def __init__(self, val_data):
        self.val_data = val_data
    
    def on_train_begin(self, logs={}):
        self.train = []
        self.val = []

    def on_epoch_end(self, batch, logs={}):
        self.train.append([logs.get('loss'), logs.get('acc')])
        x, y = self.val_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        self.val.append([loss, acc])

def plot_params(params_train, params_valid):
    plt.plot(np.array(params_train).T[1],label='Train', C='C0')
    plt.plot(np.array(params_valid).T[1],label='Validation', C='C5', alpha=0.8)
    plt.legend(loc='best')
    plt.title('Accuracy across iterations')
    plt.show()
    
    plt.plot(np.array(params_train).T[0],label='Train', C='C0')
    plt.plot(np.array(params_valid).T[0],label='Validation', C='C5', alpha=0.8)
    plt.legend(loc='best')
    plt.title('Loss across iterations')
    plt.show()