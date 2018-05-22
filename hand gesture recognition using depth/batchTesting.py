import numpy as np
from sklearn.externals import joblib
import os
import cPickle

#function to creae the confusion matrix
def update_confusion_matrix(arr, p, a):
    if p=="C":
        if a=="C":
            arr[0][0] = arr[0][0] + 1
        if a=="RA":
            arr[0][1] = arr[0][1] + 1
        if a=="RC":
            arr[0][2] = arr[0][2] + 1
        if a=="BU":
            arr[0][3] = arr[0][3] + 1
        if a=="BD":
            arr[0][4] = arr[0][4] + 1
        if a=="P":
            arr[0][5] = arr[0][5] + 1

    if p=="RA":
        if a=="C":
            arr[1][0] = arr[1][0] + 1
        if a=="RA":
            arr[1][1] = arr[1][1] + 1
        if a=="RC":
            arr[1][2] = arr[1][2] + 1
        if a=="BU":
            arr[1][3] = arr[1][3] + 1
        if a=="BD":
            arr[1][4] = arr[1][4] + 1
        if a=="P":
            arr[1][5] = arr[1][5] + 1
    
    if p=="RC":
        if a=="C":
            arr[2][0] = arr[2][0] + 1
        if a=="RA":
            arr[2][1] = arr[2][1] + 1
        if a=="RC":
            arr[2][2] = arr[2][2] + 1
        if a=="BU":
            arr[2][3] = arr[2][3] + 1
        if a=="BD":
            arr[2][4] = arr[2][4] + 1
        if a=="P":
            arr[2][5] = arr[2][5] + 1
    
    if p=="BU":
        if a=="C":
            arr[3][0] = arr[3][0] + 1
        if a=="RA":
            arr[3][1] = arr[3][1] + 1
        if a=="RC":
            arr[3][2] = arr[3][2] + 1
        if a=="BU":
            arr[3][3] = arr[3][3] + 1
        if a=="BD":
            arr[3][4] = arr[3][4] + 1
        if a=="P":
            arr[3][5] = arr[3][5] + 1
    
    if p=="BD":
        if a=="C":
            arr[4][0] = arr[4][0] + 1
        if a=="RA":
            arr[4][1] = arr[4][1] + 1
        if a=="RC":
            arr[4][2] = arr[4][2] + 1
        if a=="BU":
            arr[4][3] = arr[4][3] + 1
        if a=="BD":
            arr[4][4] = arr[4][4] + 1
        if a=="P":
            arr[4][5] = arr[4][5] + 1
            
    if p=="P":
        if a=="C":
            arr[5][0] = arr[5][0] + 1
        if a=="RA":
            arr[5][1] = arr[5][1] + 1
        if a=="RC":
            arr[5][2] = arr[5][2] + 1
        if a=="BU":
            arr[5][3] = arr[5][3] + 1
        if a=="BD":
            arr[5][4] = arr[5][4] + 1
        if a=="P":
            arr[5][5] = arr[5][5] + 1
        

#load the classifier from the disk
clf = joblib.load('./TrainingData/gestureclassifier.pkl')

#test[] will contain the histograms of images to be tested
test = []
#actual[] will contain the actual expressions of the corresponding histograms
actual = []

#relative path of test file
script_dir = os.path.dirname(__file__)
rel_path = script_dir + "/test_expression.txt"

test_file = rel_path
#read the file line by line

path='./BatchTesting/FeatureDepth/'
for filename in os.listdir('./BatchTesting/FeatureDepth'):
	myfile=os.path.join(path,filename)
	fv = cPickle.load(open(myfile, "r+"))
    	fv = np.array(fv)
    	fv = fv.flatten()
    	test.append(fv)
    	if 'features_clap' in filename:
        	actual.append('C')
    	if 'features_rotateanti'in filename:
        	actual.append('RA')
    	if 'features_rotateclock'in filename:
		actual.append('RC')
    	if 'features_bothup'in filename:
        	actual.append('BU')
    	if 'features_bothdown'in filename:
        	actual.append('BD')
        if 'features_palm'in filename:
        	actual.append('P')


#convert the lists to numpy arrays
test = np.array(test)
actual = np.array(actual)

#variables used to calculate number of correct and incorrect predictions
correct = 0
incorrect = 0

#create a 2D list for confusion matrix
arr = [[0]*6 for _ in range(6)]
srn = 1

#iterate over all the histograms
for i in range(0, len(test)):
    
    #predict using the loaded classifier
    prediction = clf.predict([test[i]])[0]
    
    #increase the corresponding count depending on the result being correct or incorrect
    if prediction==actual[i]:
        correct = correct + 1
    else:
        incorrect = incorrect + 1

    #update the confusion matrix
    update_confusion_matrix(arr, prediction, actual[i])
    
    #print the result
    print(srn, ". Prediction = ", prediction, "actual = ", actual[i])
            
    srn = srn+1

#print the number of correct and incorrect predictions and the confusion matrix
print("Correct = ", correct, "Incorrect = ", incorrect, "Total = ", correct+incorrect)
for i in range(6):
        print(arr[i])
