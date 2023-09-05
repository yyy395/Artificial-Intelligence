from keras.models import load_model
import numpy as np
from keras.datasets import mnist
import gc
import cv2
from pre import *
import matplotlib.pyplot as plt
from figure import figure
def tran_y(y): 
    y_ohe = np.zeros(10) 
    y_ohe[y] = 1 
    return y_ohe
def load_model_unet():
    x_pre, y_pre_hot = figure()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    y_train_hot = np.array([tran_y(y_train[i]) for i in range(len(y_train))]) 
    y_test_hot = np.array([tran_y(y_test[i]) for i in range(len(y_test))])
    y_train_hot = y_train_hot.astype('float32')
    y_test_hot = y_test_hot.astype('float32')
    permutatedData = np.zeros_like(x_train)
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
    x_train = np.concatenate((x_train, x_pre))
    y_train_hot = np.concatenate((y_train_hot, y_pre_hot))
    model = load_model('unet.hdf5')
    p_test = np.random.permutation(len(x_test))
    loss,accuracy = model.evaluate(x_test,y_test_hot)
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    for i in range(len(x_test)):
        permutatedData[i, :, :, :] = x_test[p_test[i], :, :, :] 
        permutatedLabel[i, :] = y_test_hot[p_test[i], :]
    #30个数据用于测试
    numberOfTrainingData = 30
    #x_test = permutatedData[0:numberOfTrainingData, :, :]
    #y_test_hot = permutatedLabel[0:numberOfTrainingData, :]
    x_test = x_test[0:30, :, :, :]
    y_test_hot = y_test_hot[0:30, :]
    x_test = np.concatenate((x_test, x_pre))
    y_test_hot = np.concatenate((y_test_hot, y_pre_hot))
    pre = model.predict(x_test)
    plt.figure(figsize=(30, 10))
    for i in range(30):
        plt.subplot(5, 10, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_test[i], cmap=plt.cm.binary)
        plt.xlabel(y_test_hot[i].argmax())
    plt.savefig('unet_test.png')
    r_pre = np.zeros((1,40))
    for x in range(40):
        print(pre[x])
    for x in range(40):
        r_pre[0, x] = np.array(pre[x]).argmax()
    print(r_pre)
    count_1 = 0
    for i in range(30):
        if r_pre[0, i]==np.array(y_test_hot[i, :]).argmax():
            count_1 = count_1 + 1
    acc_1 = count_1 / 30.0
    count_2 = 0
    for i in range(30, 40):
        if r_pre[0, i]==np.array(y_test_hot[i, :]).argmax():
            count_2 = count_2 + 1
    acc_2 = count_2 / 10.0
    acc = (accuracy * 10000 + count_2) / 10010.0
    with open('unet_result.txt', 'w') as file:
        #for element in r_pre:
            #file.write(str(element))
        #file.write('\n')
        file.write(str('minst_accuracy:')+str(accuracy)+'\n')
        file.write(str('自制数据_accuracy:')+str(acc_2)+'\n')
        file.write(str('accuracy:')+str(acc))
        file.close()
