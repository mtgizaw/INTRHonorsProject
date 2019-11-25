import numpy as np
import cv2
import time
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from tkinter import simpledialog

root = tk.Tk()
root.withdraw()
application_window = tk.Tk()
application_window.withdraw()
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 40
params.maxThreshold = 200
params.thresholdStep = 20
# Filter by Area.
params.filterByArea = True
params.minArea = 8
params.maxArea = 40
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.85
params.filterByConvexity = False
# Filter by Inertia
params.filterByInertia = False
# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)
# Create a descriptor generator
descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create()
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Get file from dialog
file_path = filedialog.askopenfilename(initialdir = ".",title = "Select file",filetypes = (("jpeg files","*.avi"),("all files","*.*")))
cap = cv2.VideoCapture(file_path)
#outP = open("testOutNoRound.txt","w+")
print('Enter the initial frame: ')
initialFrame = int(input()) #Start at 0
print('Enter the final frame: ')
endFrame = int(input()) #Stop at 240
#endFrame = 10;
frameNo = 0;
ret = cap.set(cv2.CAP_PROP_POS_FRAMES,frameNo)
print(ret)
kpList = [];
desc = [];
while cap.isOpened():
    if (frameNo >= endFrame): break
    ret, frame = cap.read()
    if ret:
        frame = frame[50:770,:]
    # Detect Blobs
        kpList.append(detector.detect(frame));
        keypoints = kpList[frameNo]
        keypoints, desc1 = descriptor.compute(frame,keypoints)
        desc.append(desc1);
        #while (frameNo == initialFrame) and (frameNo <= endFrame):
        frameNo = frameNo + 1
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        for k in range(0,len(keypoints)):
            cx = round(keypoints[k].pt[0])
            cy = round(keypoints[k].pt[1])
            cv2.putText(im_with_keypoints, str(k), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, .6,(0, 0, 255))
            #for frameNo in range(initialFrame + 1,endFrame):
            if (frameNo >= initialFrame+1 and frameNo <= endFrame):
                print('Frame',str(frameNo),'|',str(k),'Keypoints | Coordinates: (',str(round(keypoints[k].pt[0])),',',str(round(keypoints[k].pt[1])),')')

                windowName = 'Frame ' + str(frameNo)
                # Create the data tables first!
                cv2.imshow(windowName, im_with_keypoints)
##                if (frameNo == 1 or frameNo == 2):
##                    windowName2 = 'Classified Frame ' + str(frameNo) + '.jpg'
##                    cv2.imwrite(windowName2, frame)
                # This next step is necessary to force a draw.
                cv2.waitKey(1)
    else:
        cap.release()

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Classify blobs.
for j in range(initialFrame,endFrame-1):
    kp1 = kpList[j]
    kp2 = kpList[j+1]
    # Match descriptors.  frame 0 is train. frame 1 is query
    matches = bf.match(desc[j+1],desc[j])

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)
    #
    print('\nMatch Results \n           | Query Frame',endFrame,'   |    Train Frame',initialFrame+1,'  |')
    for k in range(0,len(matches)):
        print('Match = '+str(k)+'  | Query = '+str(matches[k].queryIdx)+'        |    Train = '+str(matches[k].trainIdx)+'       |')
    #    print([kpList[0][matches[k].queryIdx].pt,'versus',kpList[1][matches[k].trainIdx].pt])
    # Reverses match and query above
        print('           | (',str(round(kpList[j+1][matches[k].queryIdx].pt[0])),',',
                  str(round(kpList[j+1][matches[k].queryIdx].pt[1])),')    |    (',
                  str(round(kpList[j][matches[k].trainIdx].pt[0])),',',
                  str(round(kpList[j][matches[k].trainIdx].pt[1])),')   |')


