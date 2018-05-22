# hand-gesture-recognition
Python implementation of controlling vlc media player through hand gesture recognition using depth camera.
<p>Gestures used are:<p/>
<p>Clap: Toggle<p/>
<p>Rotate anticlock: Previous<p/>
<p>Rotate Clock: Next<p/>
<p>Palm: Quit<p/><p/>
<p>Both hands up: Volume increase<p/>
<p>Boht hands down: Volume decrease<p/>

<p>Average accuracy of 95.83% is achieved in case of depth images.<p/>
<p>Average accuracy of 70.83% is achieved in case of only RGB images.<p/>


* Running the Project:
----------------------
Setting up the kinect:
1. Place kinect approximately 90-100 cm away from the person performing the gesture. 
2. There should be no obstacle between the person and the camera. 
3. The user should bring their hands forward only when they have to perform a gesture. 
4. Test the camera view using the command 'freenect-glview'. While performing the gestures, only the hand should appear red while rest of the objects appear yellowish or green.

For Training:
1. Run the file 'newkinect.py' using the command 'python newkinect.py'. Wait for a few seconds until a sequence of zeroes starts printing on the terminal and perform the gesture in front of kinect camera. This will generate depth thresholded images and save it in the specified subject and gesture folder. Change the Subject folder and gesture folder in line no. 127 of the code. For example: For 'subject 1' and gesture 'Palm', the address will be './newdata/Depth/Depthpalm/Sub1/1palm{:d}.jpg'. 

2. Perform feature extraction by running the file 'featureExtraction.py' using the command 'python featureExtraction.py -featuresDir ./newdata/Depth/Depthpalm/Sub1 palm1' The third argument will be the path of the folder containing the depth images whose feature vector has to be generated. The fourth argument will be the name of the output file in which the feature vectors will be saved. 

3. Repeat steps 1 and 2 for each gesture of each subject.

4. Train the classifier by running the file 'training.py' using the command 'python training.py'.

For real-time testing:
1. To perform real-time testing, run the program 'realTimeTesting.py' using the command 'python realTimeTesting.py'.

For batch testing:
1. Repeat steps 1 and 2 of training phase to collect data for batch testing. Instead of folder name 'newdata' the folder should be 'BatchTesting'.

2. Run the file 'batchTesting.py' using the command 'python batchTesting.py'.
