import cv2
import time
import os
print(cv2.__version__)
dispW=640
dispH=480
flip=0
'''
camSet='nvarguscamerasrc sensor-id=1  !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method='+str(flip)+' ! video/x-raw, width='+str(dispW)+', height='+str(dispH)+', format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink'
cam= cv2.VideoCapture(camSet)
'''
i=0
j=1

cam= cv2.VideoCapture(2)

while True:
    ret,frame=cam.read()
    cv2.imshow('nanoCam',frame)
    if i==50 :
        print("即将拍照......")
    if i==50 :
        print("已经拍"+str(j)+"张")
        cv2.imwrite(str(j)+'.png',frame)
        j=1+j
        i=0
    i=1+i
    #print(i)
    #time.sleep(5) 
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
    



