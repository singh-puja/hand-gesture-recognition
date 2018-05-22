import numpy as np
import sys, time, glob, os, ntpath, cv2, numpy, cPickle
import featuresHOG
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
from matplotlib import pyplot as plt
import sklearn.svm
import sklearn.decomposition
import sklearn.ensemble



def blockshaped(arr):
    blocks = []    
    blocks.append(arr[0:arr.shape[0],0:arr.shape[1]/2])
    blocks.append(arr[0:arr.shape[0],arr.shape[1]/2:-1])
    return blocks

def featureExtraction(img, PLOT = False):
    start = time.clock()
    
    blocks = blockshaped(img)
    t1 = time.clock();    #print "{0:.2f} seconds used in line detection".format(t1-start)    
       
    [fHOG1, namesHOG] = featuresHOG.getHOG(blocks[0]);            
    [fHOG2, namesHOG] = featuresHOG.getHOG(blocks[1]); 
    t6 = time.clock();    

    fv=fHOG1 + fHOG2
    fNames = namesHOG + namesHOG  

    return fv, fNames

def getFeaturesFromFile(fileName, PLOT = False):
    img = cv2.imread(fileName, cv2.CV_LOAD_IMAGE_COLOR)    # read image
    
    [F, N] = featureExtraction(img, PLOT)
    #print(F)            # feature extraction
    return F, N

def getFeaturesFromDir(dirName):
    types = ('*.jpg', '*.JPG', '*.png')    
    imageFilesList = []
    for files in types:
        imageFilesList.extend(glob.glob(os.path.join(dirName, files)))
    
    imageFilesList = sorted(imageFilesList)
    
    Features = []; 
    for i, imFile in enumerate(imageFilesList):    
        #print "{0:.1f}".format(100.0 * float(i) / len(imageFilesList))
        [F, Names] = getFeaturesFromFile(imFile)
        Features.append(F)

    Features = np.matrix(Features)

    return (Features, imageFilesList, Names)


def main(argv):
    
    if argv[1] == "-featuresDir":
        if len(argv)==4:
            (FM, Files, FeatureNames) = getFeaturesFromDir(argv[2])
            #print(FM)
            outputfileName = argv[3]
            fo = open("./newdata/FeatureDepth/Depthfeatures_"+outputfileName, "w")
            cPickle.dump(FM, fo, protocol = cPickle.HIGHEST_PROTOCOL)

            

if __name__ == '__main__':
    main(sys.argv)
