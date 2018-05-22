# import the necessary modules
import freenect
import cv2
import numpy as np
import math
import time
import webbrowser
import Queue
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
import sys
import cPickle
from multiprocessing import Pool
import glob, ntpath
import featuresHOG
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
from matplotlib import pyplot as plt
import sklearn.decomposition
import sklearn.ensemble
import Image
#framesd = Queue.Queue(20)
#framesr = Queue.Queue(20)
framesd = []
framesr = []
fd1 = []
fr1 = []
fd2 = []
fr2 = []
fd3 = []
fr3 = []
fd4 = []
fr4 = []
#import playsound
'''working!!!'''

# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


# function to get depth image from kinect
def get_depth(flag):
    global framesd, framesr
    loopnum = 0
    i = 0
    #flag = 0
    while(True):
        print(loopnum)
        depth, _ = freenect.sync_get_depth() #depth is a numpy array which stores the depth value of each pixel captured
        rgbframes, _ = freenect.sync_get_video() #rgbframes is a numpy array which stores rgb value of each pixel captured
        rgbframes = cv2.cvtColor(rgbframes, cv2.COLOR_RGB2BGR) 
        #print(depth)
        depth_mask = np.where(depth < 650, 255, 0).astype(np.uint8)
        for i in range(depth_mask.shape[0]):
            for j in range(depth_mask.shape[1]):
                if depth_mask[i][j]==255:
                    flag = 1
                    cv2.waitKey(100)
                    break
            if flag == 1:
                break
        if flag == 1:
            framesd.append(depth_mask)
            framesr.append(rgbframes)
            loopnum = loopnum+1
            cv2.waitKey(100)
            if(loopnum==20):
                break
    #print('$$$$$')
        
    
def masking(ind):
    global fd1, fr1, fd2, fr2, fd3, fr3, fd4, fr4
    #frameNumd = 0
    #print(ind)
    if ind == 1:
        framesd = fd1
        framesr = fr1
        frameNumd = 0
        frameNumr = frameNumd
    if ind == 2:
        framesd = fd2
        framesr = fr2
        frameNumd = 5
        frameNumr = frameNumd
    if ind == 3:
        framesd = fd3
        framesr = fr3
        frameNumd = 10
        frameNumr = frameNumd
    if ind == 4:
        framesd = fd4
        framesr = fr4
        frameNumd = 15
        frameNumr = frameNumd
    m = 0
    print(framesd[m].shape)
    while(m<5):
        depth_mask=framesd[m]
        rgbframes=framesr[m]
        #print(ind)
        #print('m=', m, 'ind=', ind)
        mask = np.zeros(rgbframes.shape, np.uint8)
        a, b = depth_mask.shape
        #print('ind=', ind, 'm=', m, a, b)
        for i in range(a):
            for j in range(b):
                if(depth_mask[i][j] == 255):
                    mask[i][j][0] = 1	
                    mask[i][j][1] = 1
                    mask[i][j][2] = 1
        thresh1=depth_mask.copy() 
        
        masked_image = np.multiply(mask,rgbframes)
        print(m)
    
        #cv2.imshow('Thresholded', thresh1)
        frameNumd = frameNumd + 1
        fileName = './newData/Depth/Depthclap/Sub7/7clap{:d}.jpg'.format(frameNumd)
        cv2.imwrite(filename=fileName,img=thresh1)
        image = Image.open(fileName)
        x,y = image.size
        new_dimensions = (x/6, y/6)
        output = image.resize(new_dimensions, Image.ANTIALIAS)
        output.save(fileName, "JPEG", quality = 95)
    
        #cv2.imshow('rgbframes', rgbframes) #modified
        frameNumr = frameNumr + 1
        fileName = './newData/RGB/RGBclap/Sub7/7clap{:d}.jpg'.format(frameNumr)
        cv2.imwrite(filename=fileName,img=masked_image)
        image = Image.open(fileName)
        x,y = image.size
        new_dimensions = (x/6, y/6)
        output = image.resize(new_dimensions, Image.ANTIALIAS)
        output.save(fileName, "JPEG", quality = 95)
        m = m+1
        #cv2.waitKey(3000)
        
    
        #return (3,frameNumd, frameNumr)


if __name__ == "__main__":
    global framesd, framesr
    matrix=[0,0,0,0]
    value=2
    i=0
    flag = 0
    setq = 1
    #while(True):
    get_depth(flag)
    print("data acquired")
    jobs = [1, 11]
    num = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    fd1 = framesd[0:5]
    fd2 = framesd[5:10]
    fd3 = framesd[10:15]
    fd4 = framesd[15:20]
    fr1 = framesr[0:5]
    fr2 = framesr[5:10]
    fr3 = framesr[10:15]
    fr4 = framesr[15:20]
    p = Pool(4)
    p.map(masking, (i for i in [1, 2, 3, 4]) )

        
    
