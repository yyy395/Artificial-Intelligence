import cv2
import gc
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.models import load_model
from pre import *
from figure import figure
def tran_y(y): 
    y_ohe = np.zeros(10) 
    y_ohe[y] = 1 
    return y_ohe
def load_model_vgg():
    x_pre, y_pre_hot = figure()
    x_pre = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2BGR) for i in x_pre] 
    x_pre = np.concatenate([arr[np.newaxis] for arr in x_pre]).astype('float32')
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
    # 打乱样本数据，进行随机训练和检验
    p = np.random.permutation(len(X_train))
    for i in range(len(X_train)):
        permutatedData[i, :, :] = X_train[p[i], :, :] 
        permutatedLabel[i, :] = y_train_ohe[p[i], :]
    #5000个数据用于训练
    numberOfTrainingData = 5000
    X_train = permutatedData[0:numberOfTrainingData, :, :]
    y_train_ohe = permutatedLabel[0:numberOfTrainingData, :]
    for i in range(10):
        gc.collect()
    model = load_model('vgg16.hdf5')
    loss,accuracy = model.evaluate(X_test,y_test_ohe)
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    p_test = np.random.permutation(len(X_test))
    for i in range(len(X_test)):
        permutatedData[i, :, :, :] = X_test[p_test[i], :, :, :] 
        permutatedLabel[i, :] = y_test_ohe[p_test[i], :]
    #30个数据用于测试
    numberOfTrainingData = 30
    #x_test = permutatedData[0:numberOfTrainingData, :, :, :]
    #y_test_hot = permutatedLabel[0:numberOfTrainingData, :]
    #x_test = np.concatenate((x_test, x_pre))
    #y_test_hot = np.concatenate((y_test_hot, y_pre_hot))
    x_test = X_test[0:30, :, :, :]
    y_test_hot = y_test_ohe[0:30, :]
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
    plt.savefig('vgg_test.png')
    #plt.show()
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
    with open('vgg_result.txt', 'w') as file:
        #for element in r_pre:
            #file.write(str(element))
        #file.write('\n')
        file.write(str('minst_accuracy:')+str(accuracy)+'\n')
        file.write(str('自制数据_accuracy:')+str(acc_2)+'\n')
        file.write(str('accuracy:')+str(acc))
        file.close()