import keras
from keras.datasets import cifar10
from matplotlib import pyplot as plt
import numpy as np

def get_cifar10_data(vectorize=False):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[y_train[:,0]<2]
    y_train = y_train[y_train[:,0]<2]
    x_test = x_test[y_test[:,0]<2]
    y_test = y_test[y_test[:,0]<2]
    if vectorize:
        x_train = x_train.reshape(len(x_train), 32*32*3)
        x_test = x_test.reshape(len(x_test), 32*32*3)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = x_train*2.-1.
    x_test = x_test*2.-1.
    return (x_train, y_train), (x_test, y_test)
    
def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['Training Loss','Validation Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['Training Accuracy','Validation Accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()
    
def evaluate_test_accuracy(model,x_test,y_test):
    results = model.evaluate(x_test,y_test)
    print('Test accuracy: %0.2f%%'%(results[1]*100))