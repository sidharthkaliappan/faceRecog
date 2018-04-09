import face_recognition
import cv2
from pathlib import Path

facefolder_path="" #enter face folder path
testfolder_path="" # enter test cases path

image_paths=[str(i) for i in Path(facefolder_path).glob("*.jpg")]
test_paths=[str(i) for i in Path(testfolder_path).glob("*.jpg")]


known_face_images=[face_recognition.load_image_file(i) for i in image_paths]
known_face_encodings=[face_recognition.face_encodings(j)[0] for j in known_face_images]

known_face_names=[i.name[:-4] for i in Path(facefolder_path).glob("*.jpg")]
test_names=[i.name[:-4] for i in Path(testfolder_path).glob("*.jpg")]

total=0 # variable to track no. of faces succesfully detected
recog=0 # variable to track no. of faces succesfully recognised

for i in range(len(test_paths)):
    test_image=face_recognition.load_image_file(test_paths[i])
    test_image=cv2.resize(test_image, (0, 0), fx=0.25, fy=0.25) #resizing test_image for faster processing
    test_image=test_image[:, :, ::-1] # BGR to RGB conversion

    face_locations = []
    face_encodings = []
    face_names = []

    face_locations = face_recognition.face_locations(test_image)
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    total+=len(face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.6)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            recog+=1

        face_names.append(name)

    print("Faces found in ",test_names[i],"are : ",face_names)

print ("Total faces detected=",total)
print ("Total faces recognized=",recog)
