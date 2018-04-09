import face_recognition
import cv2

cam = cv2.VideoCapture(0)

sid_image = face_recognition.load_image_file("sid.jpeg")
sid_encode = face_recognition.face_encodings(sid_image)[0]

ret, image = cam.read()


unknown_encode = face_recognition.face_encodings(image)

result = face_recognition.compare_faces(sid_encode,unknown_encode,tolerance=0.5)[0]

print(result)
    

cam.release()
cv2.destroyAllWindows()

