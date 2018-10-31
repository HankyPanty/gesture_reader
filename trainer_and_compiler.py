from imutils.video import VideoStream
from imutils.video import FPS
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import argparse
import imutils
import time
import cv2
import subprocess as sp

x_train=[]
x_test=[]
y_train=[]
y_test=[]

####################### Creating the dataset
i=0
while(i<100):
    if(i%10):
        x_train.append(cv2.imread('data/zero/'+str(i)+'.jpeg'));y_train.append(0)
        x_train.append(cv2.imread('data/one/1'+str(i).zfill(2)+'.jpeg'));y_train.append(1)
        x_train.append(cv2.imread('data/two/2'+str(i).zfill(2)+'.jpeg'));y_train.append(2)
        x_train.append(cv2.imread('data/three/3'+str(i).zfill(2)+'.jpeg'));y_train.append(3)
        x_train.append(cv2.imread('data/four/4'+str(i).zfill(2)+'.jpeg'));y_train.append(4)
        x_train.append(cv2.imread('data/five/5'+str(i).zfill(2)+'.jpeg'));y_train.append(5)
        x_train.append(cv2.imread('data/none/6'+str(i).zfill(2)+'.jpeg'));y_train.append(6)
        i+=1
    else:
        x_test.append(cv2.imread('data/zero/'+str(i)+'.jpeg'));y_test.append(0)
        x_test.append(cv2.imread('data/one/1'+str(i).zfill(2)+'.jpeg'));y_test.append(1)
        x_test.append(cv2.imread('data/two/2'+str(i).zfill(2)+'.jpeg'));y_test.append(2)
        x_test.append(cv2.imread('data/three/3'+str(i).zfill(2)+'.jpeg'));y_test.append(3)
        x_test.append(cv2.imread('data/four/4'+str(i).zfill(2)+'.jpeg'));y_test.append(4)
        x_test.append(cv2.imread('data/five/5'+str(i).zfill(2)+'.jpeg'));y_test.append(5)
        x_test.append(cv2.imread('data/none/6'+str(i).zfill(2)+'.jpeg'));y_test.append(6)
        i+=1

######################## Normalizing the data
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
y_train=np.asarray(y_train)
y_test=np.asarray(y_test) 

model=tf.keras.models.Sequential()

######################### layers in model
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(7,activation=tf.nn.softmax))

######################### properties of model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

######################### running/training the model on x_train
model.fit(x_train,y_train, epochs=4)

######################### testing model on x_test and calculating Accuracy, prediction values.
val_loss,val_acc=model.evaluate(x_test,y_test)
print("\n\nloss    %: ",val_loss, "\naccuracy%: ",val_acc,"\n\n")
prediction=model.predict([x_test])

######################### testing for random image in x_test
i=28
print("prediction: ",np.argmax(prediction[i]))
print("actual    : ",y_test[i])
plt.imshow(x_test[i],cmap=plt.cm.binary)
plt.show()

######################### Saving the model as gesture_model
tf.keras.models.save_model(
    model,
    "final_model.model",
    overwrite=True,
    include_optimizer=True
)
print("\n\nModel SAVED\n\n")

time.sleep(3)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

ind=0;indn=0;ctr=0;
while True:
    framee=vs.read()
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
    cv2.imwrite('current.jpeg',frame)
    x_test2=[]
    x_test2.append(cv2.imread('current.jpeg'))
    x_test2=tf.keras.utils.normalize(x_test2,axis=1)
    predn=model.predict([x_test2])
    indn=np.argmax(predn[0]);per=(predn[0])[indn];
    print("prediction :", indn, "\n%surety    :", per)
    
    cv2.imshow('lol',framee)
    cv2.imshow('lol',frame)
    time.sleep(0.1)
    if(indn==ind):
        ctr+=1
        if(ctr>30):
            ctr=0
            print('\n',indn,'\n')
            val=indn
            if(val==0):
                continue;
#                 sp.call('rundll32.exe', 'user32.dll,LockWorkStation')
            elif(val==1):
                sp.call(['chrome.exe'])
            elif(val==2):
                continue;
#                 sp.call('rundll32.exe', 'user32.dll,LockWorkStation')
#                 cv2.imshow('lol',framee)
#                 cv2.imshow('lol',frame)
            elif(val==3):
                continue;
#                 sp.call('rundll32.exe', 'user32.dll,LockWorkStation')
            elif(val==4):
                break;
#                 sp.call('rundll32.exe', 'user32.dll,LockWorkStation')
            elif(val==5):
                sp.call(['rundll32.exe', 'user32.dll,LockWorkStation'])
            elif(val==6):
                print("\n\n\nHAAND NOT DETECTED\n\n");
            time.sleep(2)
    else:
        ctr=0
    ind=indn
    
vs.stop()
cv2.destroyAllWindows()