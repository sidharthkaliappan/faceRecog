import face_recognition
import cv2
from pathlib import Path
################################################################################

facefolder_path="" #enter known faces folder
testfolder_path="" #enter testcase folder
resultfolder_path="" #enter path for images to be stored

Tolerance=0.6
scaling_factor=0.25
hog_model= ("hog",1)
cnn_model=("cnn",0)

detection_model= hog_model      #SELECT FACE DETECTION MODEL

################################################################################

image_paths_names=[[str(i),i.name[:-4]] for i in Path(facefolder_path).glob("*.jpg")]
test_paths_names=[[str(i),i.name[:-4]] for i in Path(testfolder_path).glob("*.jpg")]

known_face_encodings=[face_recognition.face_encodings(face_recognition.load_image_file(i[0]))[0] for i in image_paths_names]

total=0 # variable to track no. of faces succesfully detected
recog=0 # variable to track no. of faces succesfully recognised

################################################################################
def save_results():

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations
        top *= int(1/scaling_factor)
        right *= int(1/scaling_factor)
        bottom *= int(1/scaling_factor)
        left *= int(1/scaling_factor)

        # Draw a box around the face
        cv2.rectangle(test_og_image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(test_og_image, (left, bottom - 27), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_TRIPLEX
        cv2.putText(test_og_image, name, (left + 10, bottom - 7), font, 1.0, (0, 0, 0), 1)

    # Display the resulting image
    cv2.imwrite(resultfolder_path+'result_'+test_paths_names[i][1]+".jpg", test_og_image)

################################################################################

for i in range(len(test_paths_names)):
    test_og_image=cv2.imread(test_paths_names[i][0])
    test_image=cv2.resize(test_og_image, (0, 0), fx=scaling_factor, fy=scaling_factor) #resizing test_image for faster processing
    test_image=test_image[:, :, ::-1] # BGR to RGB conversion

    face_locations = []
    face_encodings = []
    face_names = []

    face_locations = face_recognition.face_locations(test_image,number_of_times_to_upsample=detection_model[1], model=detection_model[0])
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    total+=len(face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=Tolerance)
        name = "Unknown"
        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = image_paths_names[first_match_index][1]
            recog+=1
        face_names.append(name)

    print("Faces found in ",test_paths_names[i][1],"are : ",face_names)
    save_results()


print ("Total faces detected=",total)
print ("Total faces recognized=",recog)
