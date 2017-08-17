#混合高斯建模
import numpy as np
import cv2
cap = cv2.VideoCapture('viptraffic.avi')
fourcc = cv2.cv.FOURCC(*'XVID')
out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (160,120),0)  #写视频的参数设置
fgbg=cv2.BackgroundSubtractorMOG(200,5,0.7)   #创建一个背景对象
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.namedWindow('fgmask', cv2.WINDOW_NORMAL)   #创建两个窗口
i=0
while(1):
    ret,frame = cap.read()     #读入视频帧
    i=i+1
    print i       #统计帧数
    if ret:       #读入正确则继续
        fgmask=fgbg.apply(frame,learningRate=0.02)   #计算前景掩模
        out.write(fgmask)     #写入视频
        cv2.imshow('img',frame) 
        cv2.imshow('fgmask',fgmask)    #显示结果
        k=cv2.waitKey(1000)
        if k==27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
