# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:33:47 2017

@author: i13yamamoto2y
"""

import cv2

imgpath = "C:/Users/i13yamamoto2y/Documents/tmp/camera{0}/{1}.jpg"
imgpath2 = r"C:\Users\i13yamamoto2y\Documents\tmp\stereo\{0}.jpg"

cap = cv2.VideoCapture(0);
cap2 = cv2.VideoCapture(1);
i = 0;
while(True):
    ret, frame = cap.read();
    ret, frame2 = cap2.read();
    cv2.imshow("image", frame);
    cv2.imshow("image2", frame2);
    
    key = cv2.waitKey(1);
    if key == ord("a"):
        break;
    if key == ord("s"):
        #cv2.imwrite(imgpath.format(1, i), frame);
        #cv2.imwrite(imgpath.format(2, i), frame2);
        
        cv2.imwrite(imgpath2.format(i), frame);
        cv2.imwrite(imgpath2.format(i+1), frame2);
        
        i += 1;


cap.release();
cap2.release();
cv2.destroyAllWindows();
