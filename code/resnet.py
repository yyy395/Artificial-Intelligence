from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from contextlib import redirect_stdout
import cv2
def tran_y(y): 
    y_ohe = np.zeros(10) 
    y_ohe[y] = 1 
    return y_ohe
def resnet():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))

    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    y_train_hot = np.array([tran_y(y_train[i]) for i in range(len(y_train))]) 
    y_test_hot = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
    y_train_hot = y_train_hot.astype('float32')
    y_test_hot = y_test_hot.astype('float32')
    permutatedData = np.zeros_like((x_train))
    permutatedLabel = np.zeros_like(y_train_hot)
    # 打乱样本数据，进行随机训练和检验
    p = np.random.permutation(len(x_train))
    for i in range(len(x_train)):
        permutatedData[i, :, :, :] = x_train[p[i], :, :, :] 
        permutatedLabel[i, :] = y_train_hot[p[i], :]
    #5000个数据用于训练
    numberOfTrainingData = 5000
    x_train = permutatedData[0:numberOfTrainingData, :, :]
    y_train_hot = permutatedLabel[0:numberOfTrainingData, :]
    input_img = Input(shape=(28, 28, 1))
    conv1 = Conv2D(32, 3, padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv2)
    added_1 = Add()([conv2, pool1])
    conv3 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(added_1)
    conv3 = Conv2D(32, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv3)
    added_2 = Add()([conv3, added_1])
    down_1 = Conv2D(64, 1, padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(added_2)
    conv4 = Conv2D(64, 3, padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(added_2)
    conv4 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv4)
    added_3 = Add()([conv4, down_1])
    conv5 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(added_3)
    conv5 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv5)
    added_4 = Add()([conv5, added_3])
    down_2 = Conv2D(128, 1, padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(added_4)
    conv6 = Conv2D(128, 3, padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(added_4)
    conv6 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)
    added_5 = Add()([conv6, down_2])
    conv7 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(added_5)
    conv7 = Conv2D(128, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)
    added_6 = Add()([conv7, added_5])
    down_3 = Conv2D(256, 1, padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(added_6)
    conv8 = Conv2D(256, 3, padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(added_6)
    conv8 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)
    added_7 = Add()([conv8, down_3])
    conv9 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(added_7)
    conv9 = Conv2D(256, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv9)
    added_8 = Add()([conv8, added_7])
    down_4 = Conv2D(512, 1, padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(added_8)
    conv10 = Conv2D(512, 3, padding='same', strides=(2, 2), activation='relu', kernel_initializer='he_normal')(added_8)
    conv10 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv10)
    added_9 = Add()([conv10, down_4])
    conv11 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(added_9)
    conv11 = Conv2D(512, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv11)
    added_10 = Add()([conv11, added_9])
    pool2 = GlobalAveragePooling2D()(added_10)
    pool_flat = Flatten()(pool2)
    fc1 =Dense(512, activation='relu')(pool_flat)
    fc2 =Dense(10, activation='softmax')(fc1)
    model = Model(input_img, fc2)
    learning_rate = 1e-4
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    with open('ModelSummary_resnet.txt', 'w') as f:
        with redirect_stdout(f):
            model.summary(line_length=300, positions=[.33, .55, .67, 1.])
    filepath = "resnet.hdf5"
    model_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, period=2)
    history = model.fit(x_train, y_train_hot, epochs=10, batch_size=10, shuffle=True, validation_data=(x_test, y_test_hot), callbacks=[model_checkpoint],)
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('resnet_loss.png')
    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('resnet_accuracy.png')

    plt.figure(figsize=(30, 10))
    for i in range(30):
        plt.subplot(5, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.xlabel(y_test[i])
    plt.savefig('resnet_test.png')
    
    