import numpy as np
from PIL import Image

def pre_pic(picName):
    all_values=picName.split('.')
    correct_label=int(all_values[0])
    print(correct_label,"correct label")
    img=Image.open('../number/' + picName)  #读取图像
    reIm=img.resize((28,28),Image.ANTIALIAS)  #将图像大小变成28*28
    im2_arr=np.array(reIm.convert('L'))
    for i in range(28):
        for j in range(28):
            im2_arr[i][j]=255-im2_arr[i][j]   #模型要求黑底白字，输入图为白底黑字，对每个像素点的值改为255-原值=互补的反色
 
    nm_arr=im2_arr.astype(np.float32)
    img_ready=np.multiply(nm_arr,1.0/255.0)   #从0-255之间的数变为0-1之间的浮点数
 
    return img_ready,correct_label