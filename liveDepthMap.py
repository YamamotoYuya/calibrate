# -*- coding: utf-8 -*-
"""
Created on Thu May 11 14:47:03 2017

@author: i13yamamoto2y
"""


import xml.etree.ElementTree as ET
import cv2
import numpy as np
import time

xmlpath = r"C:\Users\i13yamamoto2y\Documents\tmp\camera{0}\matrix.xml"
imgpath = r"C:\Users\i13yamamoto2y\Documents\tmp\stereo\{0}"

#read cameramatrix1 from xml
tree = ET.parse(xmlpath.format(1));
root = tree.getroot();

cameraMatrix1 = np.zeros((3, 3), np.float32);
for i in range(3):
    for j in range(3):
        string = root.findtext(".//camera/cameraMatrix/col{0}/row{1}".format(i, j));
        cameraMatrix1[i][j] = float(string);

#read distCoeffs1 from xml
distCoeffs1 = np.zeros((1, 5), np.float32);
for i in range(5):
    string = root.findtext(".//camera/distCoeffs/row{0}".format(i));
    distCoeffs1[0][i] = float(string);


#read cameramatrix2 from xml
tree = ET.parse(xmlpath.format(2));
root = tree.getroot();

cameraMatrix2 = np.zeros((3, 3), np.float32);
for i in range(3):
    for j in range(3):
        string = root.findtext(".//camera/cameraMatrix/col{0}/row{1}".format(i, j));
        cameraMatrix2[i][j] = float(string);

#read distCoeffs2 from xml
distCoeffs2 = np.zeros((1, 5), np.float32);
for i in range(5):
    string = root.findtext(".//camera/distCoeffs/row{0}".format(i));
    distCoeffs2[0][i] = float(string);


cap = cv2.VideoCapture(0);
cap2 = cv2.VideoCapture(1);

orb = cv2.ORB_create();



while(True):
    pts1 = [];
    pts2 = [];    
    
    key = cv2.waitKey(1);
    if key == ord("a"):
      break;
        
    #load and calibrate image                    
    ret, img1 = cap.read();
    img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY);
    ret, img2 = cap2.read();
    img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY);
    
    height, width = img1.shape[:2];
    
    imgL = cv2.undistort(img1, cameraMatrix1, distCoeffs1, None, None);
    imgR = cv2.undistort(img2, cameraMatrix2, distCoeffs2, None, None);
    
    
    
    #calc match points
    kp1, des1 = orb.detectAndCompute(imgL, None);
    kp2, des2 = orb.detectAndCompute(imgR, None);
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, True);
    matches = bf.match(des1, des2);
    matches = sorted(matches,key = lambda x:x.distance);
    
    if len(matches) == 0:
        print(matches);
        
    for mat in matches:
        idx1 = mat.queryIdx;
        idx2 = mat.trainIdx;
        pts1.append(kp1[idx1].pt);
        pts2.append(kp2[idx2].pt);
    pts1 = np.int32(pts1);
    pts2 = np.int32(pts2);
    

    #calc fundamental matrix
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS);#ロバスト推定
    ret, HL, HR = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (width,height));
    dstR = (imgL, HL, (width,height));
    dstL = (imgR, HR, (width,height));
    #stereoBM
    stereo = cv2.StereoBM_create(16);
    display = stereo.compute(dstL[0], dstR[0]);
    cv2.imshow("imageL", imgL);
    cv2.imshow("imageR", imgR);
    cv2.imshow("depth", display);

    time.sleep(0.5);
    
cap.release();
cap2.release();
cv2.destroyAllWindows();


#cv2.imshow("hoge", display);
#cv2.waitKey(0);
#cv2.destroyAllWindows();