# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:43:16 2017

@author: i13yamamoto2y
"""

import cv2

cap2 = cv2.VideoCapture(1);
i = 0;
while(True):
    ret, frame2 = cap2.read();
    
    cv2.imshow("image2", frame2);
    
    key = cv2.waitKey(1)
    if key == ord("a"):
        break;
    if key == ord("s"):
        cv2.imwrite("C:/Users/i13yamamoto2y/Documents/tmp/camera2/{0}.jpg".format(i), frame2);
        i += 1;

cap2.release();
cv2.destroyAllWindows();
