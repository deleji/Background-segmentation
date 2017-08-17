#单高斯背景建模
import numpy as np
import cv2
cap = cv2.VideoCapture('viptraffic.avi')
ret,frame = cap.read()
frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
frame1=cv2.resize(frame1,(256,256),interpolation=cv2.INTER_AREA)

#参数设置
learningRate = 0.03  #学习率
std_init = 20  #初始化标准差
var_init =  std_init * std_init  #初始化方差
lamda = 2.5*1.2 ;  #背景更新参数
output=np.zeros_like(frame1)  #输出图像

#初始化背景模型
frame_u=frame1    #均值模型
frame_std=np.ones(frame1.shape,dtype=np.float32)*std_init    #标准差模型
frame_var=np.ones(frame1.shape,dtype=np.float32)*var_init    #方差模型

#检测前景像素和背景像素
while(1):
    ret,frame=cap.read()
    frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)               #转换为灰度图
    frame1=cv2.resize(frame1,(256,256),interpolation=cv2.INTER_AREA)        #转换图像尺寸
    for i in range(256):
        for j in range(256):
            if(abs(int(frame1[i,j])-int(frame_u[i,j]))<lamda*frame_std[i,j]):        #判断是否为背景
               output[i,j]=0
               frame_u[i,j]=(1-learningRate)*frame_u[i,j]+learningRate*frame1[i,j]     #背景更新
               frame_var[i,j]=(1-learningRate)*frame_var[i,j]+learningRate*(int(frame1[i,j])-int(frame_u[i,j]))**2
               frame_std[i,j]=pow(frame_var[i,j],0.5)
            else:
                output[i,j]=255
    cv2.imshow('fgmask',output)              #显示前景掩膜
    cv2.imshow('background',frame_u)         #显示背景
    k=cv2.waitKey(30)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
    
