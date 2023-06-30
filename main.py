import cv2
import numpy as np
import face_recognition

imgran= face_recognition.load_image_file('Images/ranveer-singh.jpg')
imgran= cv2.cvtColor(imgran,cv2.COLOR_BGR2RGB)
imgtest= face_recognition.load_image_file('Images/ranveer-testing.jpg')
imgtest= cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

############-----Face Detecting-----------##############
faceLoc= face_recognition.face_locations(imgran)[0]
encoderan= face_recognition.face_encodings(imgran)[0]
cv2.rectangle(imgran,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest= face_recognition.face_locations(imgtest)[0]
encoderantest= face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)
###########--------------------------------##############


#compairing the faces by 128 types marks of faces and showing up the results whether images match or not
results= face_recognition.compare_faces([encoderan],encoderantest)

#now if there are some images present having same specs so
#####-------For Finding the Best Match-------####

facedis= face_recognition.face_distance([encoderan],encoderantest)
#lower distance = best match

print(results,facedis)
cv2.putText(imgtest,f'{results} {round(facedis[0],2)}',(50,50),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)

# print(faceLoc)



cv2.imshow('Ranveer Singh',imgran)
cv2.imshow('Ranveer Singh Testing IMG',imgtest)

cv2.waitKey(0)
