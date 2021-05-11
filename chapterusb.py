import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from PIL import Image



def undistort2(frame, K, D, DIM, scale=0.6, imshow=False):
    img = frame
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


cam2= cv2.VideoCapture(2)




while True:

    ret,frame=cam2.read()
  

    DIM=(1920, 1080)
    K=np.array([[602.2489991287233, 0.0, 970.1559743381282], [0.0, 601.5820354273767, 514.6547665735163], [0.0, 0.0, 1.0]])
    D=np.array([[-0.22620160841173992], [0.07654222060971734], [-0.060105741030695], [0.02295267373483664]])

    img2 = undistort2(frame, K, D, DIM)
   
    cv2.imshow("usb",img2)





    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
    






