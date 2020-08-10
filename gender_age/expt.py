'''
This file is for experimenting. I used it to check out whether images and labels are correctly stored in the HDf5 files
'''
import cv2
import numpy as np
import h5py
import imutils

db = h5py.File("/home/hrushikesh/images/adience/hdf5/age_val.hdf5","r")
print(db.keys())

#print(db["images"][3] == db["images"][5])

for i in [33, 71, 64, 32, 2, 16, 19, 14, 51, 63, 5]:
    image = np.uint8(db["images"][i])
    image = imutils.resize(image, width=200, height=180)
    print(db["labels"][i])
    cv2.imshow("image", image)
    cv2.waitKey(0)

'''
(0, 2) => 0
(4, 6) => 5
(8, 12) => 7, 1
(15, 20) =>1 
(25, 32) => 2
(38, 43) =>3
(48, 53) =>4
(60, 100) =>
'''
