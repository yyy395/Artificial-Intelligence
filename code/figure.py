from pre import *
import numpy as np
def tran_y(y): 
    y_ohe = np.zeros(10) 
    y_ohe[y] = 1 
    return y_ohe
def figure():
    x_pre_1, y_pre_1 = pre_pic('6.png')
    x_pre_1 = np.reshape(x_pre_1, (1, 28, 28, 1))
    y_pre_hot_1 = tran_y(y_pre_1)
    y_pre_hot_1 = y_pre_hot_1.astype('float32')
    x_pre_2, y_pre_2 = pre_pic('3.png')
    x_pre_2 = np.reshape(x_pre_2, (1, 28, 28, 1))
    y_pre_hot_2 = tran_y(y_pre_2)
    y_pre_hot_2 = y_pre_hot_2.astype('float32')
    x_pre = np.concatenate((x_pre_1, x_pre_2))
    y_pre_hot = np.array(np.concatenate((np.mat(y_pre_hot_1), np.mat(y_pre_hot_2))))
    x_pre_3, y_pre_3 = pre_pic('1.png')
    x_pre_3 = np.reshape(x_pre_3, (1, 28, 28, 1))
    y_pre_hot_3 = tran_y(y_pre_3)
    y_pre_hot_3 = y_pre_hot_3.astype('float32')
    x_pre = np.concatenate((x_pre, x_pre_3))
    y_pre_hot = np.array(np.concatenate((np.mat(y_pre_hot), np.mat(y_pre_hot_3))))
    x_pre_4, y_pre_4 = pre_pic('2.png')
    x_pre_4 = np.reshape(x_pre_4, (1, 28, 28, 1))
    y_pre_hot_4 = tran_y(y_pre_4)
    y_pre_hot_4 = y_pre_hot_4.astype('float32')
    x_pre = np.concatenate((x_pre, x_pre_4))
    y_pre_hot = np.array(np.concatenate((np.mat(y_pre_hot), np.mat(y_pre_hot_4))))
    x_pre_5, y_pre_5 = pre_pic('5.png')
    x_pre_5 = np.reshape(x_pre_5, (1, 28, 28, 1))
    y_pre_hot_5 = tran_y(y_pre_5)
    y_pre_hot_5 = y_pre_hot_5.astype('float32')
    x_pre = np.concatenate((x_pre, x_pre_5))
    y_pre_hot = np.array(np.concatenate((np.mat(y_pre_hot), np.mat(y_pre_hot_5))))
    x_pre_6, y_pre_6 = pre_pic('8.png')
    x_pre_6 = np.reshape(x_pre_6, (1, 28, 28, 1))
    y_pre_hot_6 = tran_y(y_pre_6)
    y_pre_hot_6 = y_pre_hot_6.astype('float32')
    x_pre = np.concatenate((x_pre, x_pre_6))
    y_pre_hot = np.array(np.concatenate((np.mat(y_pre_hot), np.mat(y_pre_hot_6))))
    x_pre_7, y_pre_7 = pre_pic('4.png')
    x_pre_7 = np.reshape(x_pre_7, (1, 28, 28, 1))
    y_pre_hot_7 = tran_y(y_pre_7)
    y_pre_hot_7 = y_pre_hot_7.astype('float32')
    x_pre = np.concatenate((x_pre, x_pre_7))
    y_pre_hot = np.array(np.concatenate((np.mat(y_pre_hot), np.mat(y_pre_hot_7))))
    x_pre_8, y_pre_8 = pre_pic('7.png')
    x_pre_8 = np.reshape(x_pre_8, (1, 28, 28, 1))
    y_pre_hot_8 = tran_y(y_pre_8)
    y_pre_hot_8 = y_pre_hot_8.astype('float32')
    x_pre = np.concatenate((x_pre, x_pre_8))
    y_pre_hot = np.array(np.concatenate((np.mat(y_pre_hot), np.mat(y_pre_hot_8))))
    x_pre_9, y_pre_9 = pre_pic('9.png')
    x_pre_9 = np.reshape(x_pre_9, (1, 28, 28, 1))
    y_pre_hot_9 = tran_y(y_pre_9)
    y_pre_hot_9 = y_pre_hot_9.astype('float32')
    x_pre = np.concatenate((x_pre, x_pre_9))
    y_pre_hot = np.array(np.concatenate((np.mat(y_pre_hot), np.mat(y_pre_hot_9))))
    x_pre_10, y_pre_10 = pre_pic('0.png')
    x_pre_10 = np.reshape(x_pre_10, (1, 28, 28, 1))
    y_pre_hot_10 = tran_y(y_pre_10)
    y_pre_hot_10 = y_pre_hot_10.astype('float32')
    x_pre = np.concatenate((x_pre, x_pre_10))
    y_pre_hot = np.array(np.concatenate((np.mat(y_pre_hot), np.mat(y_pre_hot_10))))
    return x_pre, y_pre_hot