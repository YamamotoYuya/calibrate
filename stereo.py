# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:56:35 2017

@author: i13yamamoto2y
"""

import xml.etree.ElementTree as ET
import cv2
import numpy as np
from matplotlib import pyplot as plt


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





#load and calibrate image                    
img1 = cv2.imread(imgpath.format("0.jpg"), cv2.IMREAD_GRAYSCALE);
img2 = cv2.imread(imgpath.format("1.jpg"), cv2.IMREAD_GRAYSCALE);

height, width = img1.shape[:2];

imgL = img1#cv2.undistort(img1, cameraMatrix1, distCoeffs1, None, None);
imgR = img2#cv2.undistort(img2, cameraMatrix2, distCoeffs2, None, None);

#calc match points
orb = cv2.ORB_create();
kp1, des1 = orb.detectAndCompute(imgL, None);
kp2, des2 = orb.detectAndCompute(imgR, None);

bf = cv2.BFMatcher(cv2.NORM_HAMMING, True);
matches = bf.match(des1, des2);
matches = sorted(matches,key = lambda x:x.distance);

pts1 = [];
pts2 = [];
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
dstL = cv2.warpPerspective(imgL, HL, (width,height));
dstR = cv2.warpPerspective(imgR, HR, (width,height));

#stereoBM
stereo = cv2.StereoBM_create(16);
disp = stereo.compute(dstL, dstR);

orb = cv2.ORB_create();
kp1, des1 = orb.detectAndCompute(dstL, None);
kp2, des2 = orb.detectAndCompute(dstR, None);

bf = cv2.BFMatcher(cv2.NORM_HAMMING, True);
matches = bf.match(des1, des2);
matches = sorted(matches,key = lambda x:x.distance);

pts1 = [];
pts2 = [];
for mat in matches:
    idx1 = mat.queryIdx;
    idx2 = mat.trainIdx;
    pts1.append(kp1[idx1].pt);
    pts2.append(kp2[idx2].pt);
pts1 = np.int32(pts1);
pts2 = np.int32(pts2);

draw_params = dict(matchColor = (0,255,0),
                   matchesMask = None,
                   flags = 0);
displayImg = cv2.drawMatches(dstL, kp1, dstR, kp2, matches[:20], None, **draw_params);
plt.imshow(displayImg), plt.show();
cv2.imwrite(imgpath.format("matching_2.jpg"), displayImg);



cv2.imwrite(imgpath.format("streo.jpg"), disp);
cv2.imwrite(imgpath.format("streoL.jpg"), dstL);
cv2.imwrite(imgpath.format("streoR.jpg"), dstR);

'''
cv2.imshow("imgR", dstR);
cv2.imshow("imgL", dstL);
cv2.waitKey(0);
cv2.destroyAllWindows();
'''
        
    



#cv2.imshow("hoge", display);
#cv2.waitKey(0);
#cv2.destroyAllWindows();