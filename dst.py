import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from PIL import Image

i=0
j=1

def undistort0(frame0, K, D, DIM, scale=0.6, imshow=False):
    img = frame0
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == DIM[0] / DIM[
        1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0] != DIM[0]:
        img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:  # change fov
        Knew[(0, 1), (0, 1)] = scale * Knew[(0, 1), (0, 1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if imshow:
        cv2.imshow("undistorted", undistorted_img)
    return undistorted_img

def undistort1(frame1, K, D, DIM, scale=0.6, imshow=False):
    img = frame1
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    assert dim1[0] / dim1[1] == DIM[0] / DIM[
        1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0] != DIM[0]:
        img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:  # change fov
        Knew[(0, 1), (0, 1)] = scale * Knew[(0, 1), (0, 1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if imshow:
        cv2.imshow("undistorted", undistorted_img)
    return undistorted_img


def undistort2(frame1, K, D, DIM, scale=0.6, imshow=False):
    img = frame1
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort

    assert dim1[0] / dim1[1] == DIM[0] / DIM[
        1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0] != DIM[0]:
        img = cv2.resize(img, DIM, interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:  # change fov
        Knew[(0, 1), (0, 1)] = scale * Knew[(0, 1), (0, 1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if imshow:
        cv2.imshow("undistorted", undistorted_img)
    return undistorted_img



camSet0=('nvarguscamerasrc sensor-id=0 !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink')
cam0= cv2.VideoCapture(camSet0)


camSet1=('nvarguscamerasrc sensor-id=1 !  video/x-raw(memory:NVMM), width=3264, height=2464, format=NV12, framerate=21/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink')
cam1= cv2.VideoCapture(camSet1)

cam2= cv2.VideoCapture(2)




while True:
    ret,frame0=cam0.read()
    #cv2.imwrite('qianjb'+'.png',frame0)
    ret,frame1=cam1.read()
    ret,frame2=cam2.read()
  
    DIM=(640, 480)
    K=np.array([[299.6530666488737, 0.0, 313.0641975476261], [0.0, 296.02715589093975, 242.9805456633487], [0.0, 0.0, 1.0]])
    D=np.array([[-0.03374851673242447], [0.02750443523536193], [-0.048776626228677705], [0.017618450137295693]])


    K1=np.array([[296.3018421214124, 0.0, 315.2777692149451], [0.0, 293.29743208199017, 224.04310565966267], [0.0, 0.0, 1.0]])
    D1=np.array([[-0.03683663307067153], [-0.021995801782306826], [0.046911039842475376], [-0.03098652211435308]])

    DIM1=(1920, 1080)
    K2=np.array([[602.2489991287233, 0.0, 970.1559743381282], [0.0, 601.5820354273767, 514.6547665735163], [0.0, 0.0, 1.0]])
    D2=np.array([[-0.22620160841173992], [0.07654222060971734], [-0.060105741030695], [0.02295267373483664]])


    img0 = undistort0(frame0, K, D, DIM)
    #cv2.imwrite('qianqjb'+'.png',img0)
    img1 = undistort1(frame1, K1, D1, DIM)
    img2 = undistort2(frame2, K2, D2, DIM1)
    #cv2.imshow("undistort_qian",img0)
    #cv2.imshow("undistort_zuo",img1)


    #cv2.imwrite('zuo'+'.png',img1)
    #cv2.imwrite(str(j)+'.png',frame)
    #time.sleep(5) 
    #cv2.imwrite(str(j)+'.png',img)

    H_rows, W_cols= img0.shape[:2]
    #print(H_rows, W_cols)
# 原图中书本的四个角点(左上、右上、左下、右下),与变换后矩阵位置
    pts1 = np.float32([[261, 226], [372, 226], [38, 382], [619, 382]])
    pts2 = np.float32([[140, 1],[500,1],[140, 400],[500,400]])
# 生成透视变换矩阵；进行透视变换
    M0 = cv2.getPerspectiveTransform(pts1, pts2)

    pts3 = np.float32([[271, 195], [380, 195], [77, 324], [628, 324]])
    pts4 = np.float32([[140, 1],[500,1],[140, 400],[500,400]])


# 生成透视变换矩阵；进行透视变换
    M1 = cv2.getPerspectiveTransform(pts3, pts4)
    dst0 = cv2.warpPerspective(img0, M0, (640,480))
    dst1 = cv2.warpPerspective(img1, M1, (640,480))
    
    dst2 = cv2.rotate(dst1,cv2.ROTATE_90_COUNTERCLOCKWISE)
    #resized = cv2.resize(dst2, None, fx=0.6, fy=0.9, interpolation=cv2.INTER_AREA)

    #cv2.imwrite('qianqtsbh'+'.png',dst0)

    cv2.imshow("tsbh_img_qian",dst0)
    cv2.imshow("tsbh_img_zuo",dst2)
    cv2.imwrite('qianq'+'.png',dst0)
    cv2.imwrite('zuo'+'.png',dst2)

    cv2.imshow("usb",img2)








    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
    






