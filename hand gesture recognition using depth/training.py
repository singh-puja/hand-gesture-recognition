from sklearn.svm import SVC
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
import os
from sklearn.feature_extraction.text import CountVectorizer
import sys
import cPickle
reload(sys)
sys.setdefaultencoding("ISO-8859-1")
#relative path of training file
script_dir = os.path.dirname(__file__)
rel_path = script_dir + "/train_expression.txt"

train_file = rel_path

#X[] will contain the histograms of all the images
X = []
#Y[] will contain the corresponding expressions 
Y = []

#read the file line by line
path='./newdata/FeatureDepth/'
for filename in os.listdir('./newdata/FeatureDepth'):
    #extract file_name, histogram, angle and epression from the line
    print(filename)
    myfile=os.path.join(path,filename)
    
    #extract the file name, histogram, name, and expression from each line
    line = filename.split('/')
    name = line[len(line)-1]
    print(name)
    
    fv = cPickle.load(open(myfile, "r+"))
    fv = np.array(fv)
    fv = fv.flatten()
    print(fv.shape)
    
    #fv = f.read()
    #print(fv)
        
    if 'features_clap' in name:
        X.append(fv)
        Y.append('C')
    if 'features_palm'in name:
        X.append(fv)
        Y.append('P')
    if 'features_rotateanti'in name:
        X.append(fv)
        Y.append('RA')
    if 'features_rotateclock'in name:
        X.append(fv)
        Y.append('RC')
    if 'features_bothup'in name:
        X.append(fv)
        Y.append('BU')
    if 'features_bothdown'in name:
        X.append(fv)
        Y.append('BD')
        
#convert both the lists to numpy arrays
X = np.array(X)
Y = np.array(Y)


#print(X.shape)
#print(Y.shape)
#build the classifier
clf = OneVsRestClassifier(SVC(kernel='linear'))
clf.fit(X, Y)

#store the constructed classifier on the disk
joblib.dump(clf, './TrainingData/gestureclassifier.pkl')
