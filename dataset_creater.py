from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

##########creating database

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
# fps = FPS().start()
i=0;

while True:
    
    framee = vs.read()
#     framee = imutils.resize(frame, width=400)
#     fps.update()
    framee=cv2.flip(framee,1)
    roi=framee[100:300, 100:300]
    cv2.rectangle(framee,(100,100),(300,300),(0,255,0),0)    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0,20,70], dtype=np.uint8)
    upper_skin = np.array([20,255,255], dtype=np.uint8)
    frame = cv2.inRange(hsv, lower_skin, upper_skin)
    
    kernel = np.ones((3,3),np.uint8)
    frame = cv2.dilate(frame,kernel,iterations = 7)
    frame = cv2.erode(frame,kernel,iterations = 3)
    frame = cv2.GaussianBlur(frame,(5,5),100)
    frame = cv2.resize(frame,(50,50))
    
    cv2.imshow('lol',framee)
    cv2.imshow('lol',frame)
    
    if(i==0 or i==100 or i==200 or i==300 or i==400 or i==500 or i==600):
        print("next stage")
        time.sleep(5)
    if(i<100):
        locn="data/zero/"+str(i)+".jpeg"
        cv2.imwrite(locn,frame)
    elif(i<200):
        locn="data/one/"+str(i)+".jpeg"
        cv2.imwrite(locn,frame)
    elif(i<300):
        locn="data/two/"+str(i)+".jpeg"
        cv2.imwrite(locn,frame)
    elif(i<400):
        locn="data/three/"+str(i)+".jpeg"
        cv2.imwrite(locn,frame)
    elif(i<500):
        locn="data/four/"+str(i)+".jpeg"
        cv2.imwrite(locn,frame)
    elif(i<600):
        locn="data/five/"+str(i)+".jpeg"
        cv2.imwrite(locn,frame)
    elif(i<700):
        locn="data/none/"+str(i)+".jpeg"
        cv2.imwrite(locn,frame)
#     elif(i<800):
#         locn="data/nice/"+str(i)+".jpeg"
#         cv2.imwrite(locn,frame)
    if(i>700):
        break;
    time.sleep(0.05)
#     print(i)
    i+=1

# fps.stop()
vs.stop()
# print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()