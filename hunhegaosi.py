#��ϸ�˹��ģ
import numpy as np
import cv2
cap = cv2.VideoCapture('viptraffic.avi')
fourcc = cv2.cv.FOURCC(*'XVID')
out = cv2.VideoWriter('output1.avi',fourcc, 20.0, (160,120),0)  #д��Ƶ�Ĳ�������
fgbg=cv2.BackgroundSubtractorMOG(200,5,0.7)   #����һ����������
cv2.namedWindow('img', cv2.WINDOW_NORMAL)
cv2.namedWindow('fgmask', cv2.WINDOW_NORMAL)   #������������
i=0
while(1):
    ret,frame = cap.read()     #������Ƶ֡
    i=i+1
    print i       #ͳ��֡��
    if ret:       #������ȷ�����
        fgmask=fgbg.apply(frame,learningRate=0.02)   #����ǰ����ģ
        out.write(fgmask)     #д����Ƶ
        cv2.imshow('img',frame) 
        cv2.imshow('fgmask',fgmask)    #��ʾ���
        k=cv2.waitKey(1000)
        if k==27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()
