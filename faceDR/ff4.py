import face_recognition
import cv2
from pathlib import Path
################################################################################

facefolder_path="faces/faces/" #enter known faces folder
testfolder_path="test images/test images" #enter testcase folder
resultfolder_path="store/" #enter path for images to be stored

Tolerance=0.6
scaling_factor=1
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

    for (top, right, bottom, left), name , percent in zip(face_locations, face_names, face_percents):
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
        info=name+"("+str(percent)+" %)"
        cv2.putText(test_og_image, info, (left + 10, bottom - 7), font, 1.0, (0, 0, 0), 1,cv2.LINE_AA)

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
    face_percents=[]

    face_locations = face_recognition.face_locations(test_image,number_of_times_to_upsample=detection_model[1], model=detection_model[0])
    face_encodings = face_recognition.face_encodings(test_image, face_locations)

    total+=len(face_locations)

    for face_encoding in face_encodings:

        name = "Unknown"
        percent = 0

        distances=list(face_recognition.face_distance(known_face_encodings, face_encoding))
        least=min(distances)
        check = (1-least)*100  # deriving % matching

        if check>((1-Tolerance)*100): # recognize only above tolerable matching %
            name = image_paths_names[distances.index(least)][1] # getting corresponding names
            percent = round(check,2) # rounding off for better representation
            recog+=1

        face_names.append(name)
        face_percents.append(percent)


    print("Faces found in ",test_paths_names[i][1],"are : ",face_names,face_percents)
    save_results()

################################################################################

print ("Total faces detected=",total)
print ("Total faces recognized=",recog)
