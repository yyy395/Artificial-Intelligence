from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from contextlib import redirect_stdout
from pre import *

def tran_y(y): 
    y_ohe = np.zeros(10) 
    y_ohe[y] = 1 
    return y_ohe
def unet():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    x_train = x_train[0:5000, :, :, :]
    y_train = y_train[0:5000]
    y_train_hot = np.array([tran_y(y_train[i]) for i in range(len(y_train))]) 
    y_test_hot = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
    y_train_hot = y_train_hot.astype('float32')
    y_test_hot = y_test_hot.astype('float32')
    permutatedData = np.zeros_like(x_train)
    permutatedLabel = np.zeros_like(y_train_hot)
    p = np.random.permutation(len(x_train))
    for i in range(len(x_train)):
        permutatedData[i, :, :, :] = x_train[p[i], :, :, :] 
        permutatedLabel[i, :] = y_train_hot[p[i], :]
    numberOfTrainingData = 5000
    x_train = permutatedData[0:numberOfTrainingData, :, :]
    y_train_hot = permutatedLabel[0:numberOfTrainingData, :]
    input_img = Input(shape=(28, 28, 1))
    conv1 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(input_img)
    conv1 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    conv1 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    up2 = Conv2DTranspose(32, 2, strides=(2, 2), padding='same', activation='relu', use_bias=False, kernel_initializer='he_normal')(conv3)
    merge2 = concatenate([conv2, up2], axis=-1)
    conv4 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge2)
    conv4 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    up1 = Conv2DTranspose(16, 2, strides=(2, 2), padding='same', activation='relu', use_bias=False, kernel_initializer='he_normal')(conv4)
    merge1 = concatenate([conv1, up1], axis=-1)
    conv5 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge1)
    conv5 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    conv5 = Conv2D(1, 1, padding='same', activation=None, kernel_initializer='he_normal')(conv5)
    added = Add()([conv5, input_img])
    added_flat = Flatten()(added)
    fc1 =Dense(32, activation='relu')(added_flat)
    fc2 =Dense(10, activation='softmax')(fc1)
    model = Model(input_img, fc2)
    learning_rate = 1e-4
    model.compile(loss='huber', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    with open('ModelSummary_unet.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary(line_length=300, positions=[.33, .55, .67, 1.])
    filepath = "unet.hdf5"
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, period=2)
    history = model.fit(x_train, y_train_hot, epochs=10, batch_size=10, shuffle=True, validation_data=(x_test, y_test_hot), callbacks=[model_checkpoint],)

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('unet_loss.png')
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('unet_accuracy.png')