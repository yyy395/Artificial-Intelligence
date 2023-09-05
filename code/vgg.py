import numpy as np
from keras.datasets import mnist
import gc
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import cv2
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from figure import figure
##由于输入层需要10个节点，所以最好把目标数字0-9做成one Hot编码的形式。
def tran_y(y): 
    y_ohe = np.zeros(10) 
    y_ohe[y] = 1 
    return y_ohe
def vgg16():
    ishape = 48
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_train] 
    X_train = np.concatenate([arr[np.newaxis] for arr in X_train]).astype('float32') 
    X_train /= 255.0
    X_test = [cv2.cvtColor(cv2.resize(i, (ishape, ishape)), cv2.COLOR_GRAY2BGR) for i in X_test] 
    X_test = np.concatenate([arr[np.newaxis] for arr in X_test]).astype('float32')
    X_test /= 255.0
    y_train_ohe = np.array([tran_y(y_train[i]) for i in range(len(y_train))]) 
    y_test_ohe = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
    y_train_ohe = y_train_ohe.astype('float32')
    y_test_ohe = y_test_ohe.astype('float32')
    permutatedData = np.zeros_like((X_train))
    permutatedLabel = np.zeros_like(y_train_ohe)
    p = np.random.permutation(len(X_train))
    for i in range(len(X_train)):
        permutatedData[i, :, :] = X_train[p[i], :, :] 
        permutatedLabel[i, :] = y_train_ohe[p[i], :]
    numberOfTrainingData = 5000
    X_train = permutatedData[0:numberOfTrainingData, :, :]
    y_train_ohe = permutatedLabel[0:numberOfTrainingData, :]
    for i in range(10):
        gc.collect()
    model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(48, 48, 3)) 
    print(model_vgg.output.shape)
    for layer in model_vgg.layers:
        layer.trainable = False
    x = Flatten()(model_vgg.output) 
    x = Dense(1024, activation='relu')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='softmax')(x) 
    model = Model(model_vgg.input, x)

    learning_rate = 1e-4
    model.compile(loss='huber', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    with open('ModelSummary_vgg16.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary(line_length=300, positions=[.33, .55, .67, 1.])
    filepath = "vgg16.hdf5"
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, period=2)
    history = model.fit(X_train, y_train_ohe, validation_data=(X_test, y_test_ohe), epochs=10, batch_size=10, callbacks=[model_checkpoint],)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('vgg_loss.png')
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('vgg_accuracy.png')
    plt.figure(figsize=(30, 10))
    for i in range(30):
        plt.subplot(5, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_test[i], cmap=plt.cm.binary)
        plt.xlabel(y_test[i])
    plt.savefig('vgg_test.png')