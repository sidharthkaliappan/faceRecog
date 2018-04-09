import face_recognition

image = face_recognition.load_image_file("sid.jpeg")
unknown_image = face_recognition.load_image_file("test5.jpg")

sid_encode = face_recognition.face_encodings(image)[0]
unknown_encode = face_recognition.face_encodings(unknown_image)[0]

results = face_recognition.compare_faces([sid_encode],unknown_encode,tolerance=0.5)
print(results)
