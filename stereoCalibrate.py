# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:22:52 2017

@author: i13yamamoto2y
"""

PATTERN_SIZE = 9;

import sys
import cv2
import numpy as np

imgpath = "C:/Users/i13yamamoto2y/Documents/tmp/camera{0}/{1}.jpg";
imgpath2 = r"C:\Users\i13yamamoto2y\Documents\tmp\stereo\{0}.jpg";

#create virtual chessboard
objPoints = np.zeros((PATTERN_SIZE*PATTERN_SIZE, 3), np.float32);
vertical, horizon = np.mgrid[0:PATTERN_SIZE, 0:PATTERN_SIZE];
objPoints[:,0] = np.array(vertical.ravel());
objPoints[:,1] = np.array(horizon.ravel());

objectPoints = [];

#read image
imgPoints1 = []
for i in range(9):
    img1 = cv2.imread(imgpath.format(1, i), cv2.IMREAD_GRAYSCALE);
    #find chessboard corners
    ret, corners = cv2.findChessboardCorners(img1, (PATTERN_SIZE, PATTERN_SIZE));
    if ret != True:
        print("cannot find corners");
        sys.exit();
    imgPoints1.append(corners);
    objectPoints.append(objPoints);

#get cameramatrix
height, width = img1.shape[:2];
ret, cameraMatrix1, distCoeffs1, rvecs1, tvecs1 = cv2.calibrateCamera(objectPoints, imgPoints1, (width, height), None, None);
#dist = cv2.undistort(img1, cameraMatrix1, distCoeffs1, None, None);


#read image
imgPoints2 = [];
for i in range(9):
    img2 = cv2.imread(imgpath.format(2, i), cv2.IMREAD_GRAYSCALE);
    #find chessboard corners
    ret, corners = cv2.findChessboardCorners(img2, (PATTERN_SIZE, PATTERN_SIZE));
    if ret != True:
        print("cannot find corners");
        sys.exit();
    imgPoints2.append(corners);

#get cameramatrix
ret, cameraMatrix2, distCoeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(objectPoints, imgPoints2, (width, height), None, None);
#dist = cv2.undistort(img2, cameraMatrix2, distCoeffs2, None, None);


img1 = cv2.imread(imgpath2.format(0), cv2.IMREAD_GRAYSCALE);
img2 = cv2.imread(imgpath2.format(1), cv2.IMREAD_GRAYSCALE);



from matplotlib import pyplot as plt
import sys
orb = cv2.sift_create();
kp1, des1 = orb.detectAndCompute(img1, None);
kp2, des2 = orb.detectAndCompute(img2, None);

bf = cv2.BFMatcher(cv2.NORM_HAMMING, True);
matches = bf.match(des1, des2);
matches = sorted(matches,key = lambda x:x.distance);
draw_params = dict(matchColor = (0,255,0),
                   matchesMask = None,
                   flags = 0);
displayImg = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None, **draw_params);
plt.imshow(displayImg), plt.show();
cv2.imwrite(imgpath2.format("streoMatching_1"), displayImg);







#ステレオ画像の平行化
ret, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(objectPoints, imgPoints1, imgPoints2, cameraMatrix1, distCoeffs1, cameraMatrix2,  distCoeffs2,
                                                (width, height));
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2,
                                                                  (width, height), R, T);                                         
mapL1, mapL2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (width, height), cv2.CV_16SC2, cv2.CV_16UC1);
mapR1, mapR2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (width, height), cv2.CV_16SC2, cv2.CV_16UC1);

dstImgL = cv2.remap(img1, mapL1, mapL2, cv2.INTER_LINEAR);
dstImgR = cv2.remap(img2, mapR1, mapR2, cv2.INTER_LINEAR);

'''
cv2.imshow("imgL", dstImgL);
cv2.imshow("imgR", dstImgR);
cv2.waitKey(0);
cv2.destroyAllWindows();
'''






orb = cv2.ORB_create();
kp1, des1 = orb.detectAndCompute(dstImgL, None);
kp2, des2 = orb.detectAndCompute(dstImgR, None);

bf = cv2.BFMatcher(cv2.NORM_HAMMING, True);
matches = bf.match(des1, des2);
matches = sorted(matches,key = lambda x:x.distance);
draw_params = dict(matchColor = (0,255,0),
                   matchesMask = None,
                   flags = 0);
displayImg = cv2.drawMatches(dstImgL, kp1, dstImgR, kp2, matches[:30], None, **draw_params);
plt.imshow(displayImg), plt.show();
cv2.imwrite(imgpath2.format("stereoMatching_2"), displayImg);


stereo = cv2.StereoSGBM_create(0, 16, 3, 21, 30, 100);
display = stereo.compute(dstImgL, dstImgR);

cv2.imwrite(imgpath2.format("streoMatching_result"), display);
cv2.imwrite(imgpath2.format("streoMatching_L"), dstImgL);
cv2.imwrite(imgpath2.format("streoMatching_R"), dstImgR);
display = display.astype(np.uint8);

cv2.imshow("imgL", dstImgL);
cv2.imshow("imgR", dstImgR);
cv2.imshow("stereo", display);
cv2.waitKey(0);
cv2.destroyAllWindows();





'''
#export xml
root1 = ET.Element("root");
camera1 = ET.SubElement(root1, "camera");
matrix1 = ET.SubElement(camera1, "cameraMatrix");

i = 0;
for idx in cameraMatrix1:
    j = 0;
    col = ET.SubElement(matrix1, "col{0}".format(i));
    for data in idx:
        row = ET.SubElement(col, "row{0}".format(j));
        row.text = str(data);
        j += 1;
    i += 1;
coeffs = ET.SubElement(camera1, "distCoeffs");

i = 0;
for idx in distCoeffs1[0]:
    row = ET.SubElement(coeffs, "row{0}".format(i));
    row.text = str(idx);
    i += 1;

tree = ET.ElementTree(root1);
tree.write(xmlpath.format(1, "matrix"));



root2 = ET.Element("root");
camera2 = ET.SubElement(root2, "camera");
matrix2 = ET.SubElement(camera2, "cameraMatrix");

i = 0;
for idx in cameraMatrix2:
    j = 0;
    col = ET.SubElement(matrix2, "col{0}".format(i));
    for data in idx:
        row = ET.SubElement(col, "row{0}".format(j));
        row.text = str(data);
        j += 1;
    i += 1;
coeffs = ET.SubElement(camera2, "distCoeffs");

i = 0;
for idx in distCoeffs2[0]:
    row = ET.SubElement(coeffs, "row{0}".format(i));
    row.text = str(idx);
    i += 1;

tree = ET.ElementTree(root2);
tree.write(xmlpath.format(2, "matrix"));
'''




