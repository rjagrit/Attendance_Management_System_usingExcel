import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

#fetching all the images in a particular folder

path= 'Attendance'
images=[]

className= []
myList= os.listdir(path)
print(myList)

for cls in myList:
    curImg= cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    className.append(os.path.splitext(cls)[0])
print(className)

def findEncodings(images):
    encodeList= []
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode= face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList= f.readlines()
        nameList= []
        for line in myDataList:
            entry= line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            now= datetime.now()
            dtString= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')





encodeListKnown= findEncodings(images)
print('Encoding Complete')

#to find matches between our encodings (Webcam image is equal to present image in Folder)

cap= cv2.VideoCapture(0)

#we doing this to reduce the size of the images for fast process
while True:
    success, img= cap.read()
    imgs= cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    facesCurFrame= face_recognition.face_locations(imgs)

    encodeCurFrame = face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches= face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDist= face_recognition.face_distance(encodeListKnown,encodeFace)
        # print(faceDist)
        matchIndex= np.argmin(faceDist)

        if matches[matchIndex]:
            name= className[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1= faceLoc
            y1, x2, y2, x1 =y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+45,y2-10),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

# faceLoc= face_recognition.face_locations(imgran)[0]
# encoderan= face_recognition.face_encodings(imgran)[0]
# cv2.rectangle(imgran,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)
#
# faceLocTest= face_recognition.face_locations(imgtest)[0]
# encoderantest= face_recognition.face_encodings(imgtest)[0]
# cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
#
# results= face_recognition.compare_faces([encoderan],encoderantest)
# facedis= face_recognition.face_distance([encoderan],encoderantest)