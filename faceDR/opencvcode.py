import cv2
import numpy
import matplotlib 
import PIL



webcam = cv2.VideoCapture(0)
y,frame = webcam.read()
#cv2.waitKey(100)
#cv2.imshow("photo",frame)
#cv2.waitKey(100)


detector = cv2.CascadeClassifier("xml/frontal_face.xml")
scale_factor = 1.2
min_neighbors = 5
min_size = (30,30)
biggest_only = True
flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
            cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
            cv2.CASCADE_SCALE_IMAGE
face_coord = detector.detectMultiScale(frame,
                                        scaleFactor=scale_factor,
                                        minNeighbors=min_neighbors,
                                        minSize=min_size,
                                        flags=flags)

print "Type: " + str(type(face_coord))
print face_coord
print "Length: "  + str(len(face_coord))



webcam.release()
