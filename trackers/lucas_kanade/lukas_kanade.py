import numpy as np
import cv2 as cv
import argparse

cap = cv.VideoCapture(cv.samples.findFile("traffic_day.mp4"))
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 50,
                       qualityLevel = 0.1,
                       minDistance = 100,
                       blockSize = 50)
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (30, 30),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

#Create Video Writer Object
#fourcc = cv.VideoWriter_fourcc('M','J','P','G')
#out = cv.VideoWriter('output2.mp4', fourcc, 24.0, (1280,  720))

n=0
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
while(1):
    
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)
    
    #Write frame
#    out.write(img)
    
    #Save image
    n=n+1
    cv.imwrite("results/out"+str(n)+".jpg", img)
    
    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
cap.release()
#out.release()
cv.destroyAllWindows()