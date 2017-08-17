#����˹������ģ
import numpy as np
import cv2
cap = cv2.VideoCapture('viptraffic.avi')
ret,frame = cap.read()
frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
frame1=cv2.resize(frame1,(256,256),interpolation=cv2.INTER_AREA)

#��������
learningRate = 0.03  #ѧϰ��
std_init = 20  #��ʼ����׼��
var_init =  std_init * std_init  #��ʼ������
lamda = 2.5*1.2 ;  #�������²���
output=np.zeros_like(frame1)  #���ͼ��

#��ʼ������ģ��
frame_u=frame1    #��ֵģ��
frame_std=np.ones(frame1.shape,dtype=np.float32)*std_init    #��׼��ģ��
frame_var=np.ones(frame1.shape,dtype=np.float32)*var_init    #����ģ��

#���ǰ�����غͱ�������
while(1):
    ret,frame=cap.read()
    frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)               #ת��Ϊ�Ҷ�ͼ
    frame1=cv2.resize(frame1,(256,256),interpolation=cv2.INTER_AREA)        #ת��ͼ��ߴ�
    for i in range(256):
        for j in range(256):
            if(abs(int(frame1[i,j])-int(frame_u[i,j]))<lamda*frame_std[i,j]):        #�ж��Ƿ�Ϊ����
               output[i,j]=0
               frame_u[i,j]=(1-learningRate)*frame_u[i,j]+learningRate*frame1[i,j]     #��������
               frame_var[i,j]=(1-learningRate)*frame_var[i,j]+learningRate*(int(frame1[i,j])-int(frame_u[i,j]))**2
               frame_std[i,j]=pow(frame_var[i,j],0.5)
            else:
                output[i,j]=255
    cv2.imshow('fgmask',output)              #��ʾǰ����Ĥ
    cv2.imshow('background',frame_u)         #��ʾ����
    k=cv2.waitKey(30)
    if k==27:
        break
cap.release()
cv2.destroyAllWindows()
    
