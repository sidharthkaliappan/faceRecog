import face_recognition
import cv2
from pathlib import Path

facefolder_path="" #enter path directory
testfolder_path="" # enter path directory

image_paths_names=[[str(i),i.name[:-4]] for i in Path(facefolder_path).glob("*.jpg")]
test_paths_names=[[str(i),i.name[:-4]] for i in Path(testfolder_path).glob("*.jpg")]

known_face_encodings=[face_recognition.face_encodings(face_recognition.load_image_file(i[0]))[0] for i in image_paths_names]

total=0 # variable to track no. of faces succesfully detected
recog=0 # variable to track no. of faces succesfully recognised

for i in range(len(test_paths_names)):
    test_image=face_recognition.load_image_file(test_paths_names[i][0])
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
            name = image_paths_names[first_match_index][1]
            recog+=1

        face_names.append(name)

    print("Faces found in ",test_paths_names[i][1],"are : ",face_names)

print ("Total faces detected=",total)
print ("Total faces recognized=",recog)
