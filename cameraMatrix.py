# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:22:52 2017

@author: i13yamamoto2y
"""

PATTERN_SIZE = 9;

import sys
import cv2
import numpy as np
import xml.etree.ElementTree as ET

imgpath = "C:/Users/i13yamamoto2y/Documents/tmp2/camera{0}/{1}.jpg";
xmlpath = "C:/Users/i13yamamoto2y/Documents/tmp/camera{0}/{1}.xml";
imgpath2 = r"C:\Users\i13yamamoto2y\Documents\tmp2\stereo\{0}.jpg";

#create virtual chessboard
objPoints = np.zeros((PATTERN_SIZE*PATTERN_SIZE, 3), np.float32);
vertical, horizon = np.mgrid[0:PATTERN_SIZE, 0:PATTERN_SIZE];
objPoints[:,0] = np.array(vertical.ravel());
objPoints[:,1] = np.array(horizon.ravel());

objectPoints = [];

#read image
imgPoints1 = []
for i in range(9):
    img1 = cv2.imread(imgpath.format(1, 10+i), cv2.IMREAD_GRAYSCALE);
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
img1 = cv2.imread("C:/Users/i13yamamoto2y/Documents/tmp/camera1/0.jpg");
distL = cv2.undistort(img1, cameraMatrix1, distCoeffs1, None, None);


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
img2 = cv2.imread("C:/Users/i13yamamoto2y/Documents/tmp/camera2/1.jpg");
distR = cv2.undistort(img2, cameraMatrix2, distCoeffs2, None, None);




cv2.imshow("imgL", distL);
cv2.imshow("imgR", distR);
cv2.waitKey(0);
cv2.destroyAllWindows();




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





