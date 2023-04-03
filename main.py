'''# encoding means an image is converted into numbers
# pip install cmake
# pip install face_recognition'''

import face_recognition
import cv2
from datetime import datetime
import csv

import numpy as np

video_capture = cv2.VideoCapture(0);

#Load Known faces

vasu_image = face_recognition.load_image_file("faces/Photo.jpg")
# Encoding - means an image is converted into numbers
vasu_encoding = face_recognition.face_encodings(vasu_image)[0]  # 0 is for face that how many faces aare there in image

# we will store their name in an array and take their name
known_face_encodings = [vasu_encoding]
known_face_names = ["Vasu"]

# list of expected students
students = known_face_names.copy() # whenever a person take his face in
# camera the face will be searched that he is the right person or not.

face_locations = []
face_encodings = []

# Get the current date and time

now = datetime.now()
current_date = now.strftime("%Y-%m-%d") # format the time


## NOW CREATE A CSV WRITER SO THAT WHO WRITE IN CSV(COMMA, SEPERATED VALUE)

f = open(f"{current_date}.csv" , "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read() # Why _, because video_capture.read first argument is that your
#  video_capture is successful or not and second argument is frame.
# now we will make a small frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy= 0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize Faceas

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings((rgb_small_frame, face_locations))

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        # tell how much the know_face_encoding is similiar with the face
        best_match_index = np.argmin(face_distance)

        if(matches[best_match_index]):
            name = known_face_names[best_match_index]   # we will get the name

            # Add the text if the person is present

            if name in known_face_names:
                font = cv2.FONT_HERSHEY_PLAIN = 1
                bottomLeftCornerOfText = (4, 100)
                fontScale = 1.0
                fontColor  = (255, 45, 55)
                thickness  = 3
                lineType = 2
                cv2.putText(frame, name + "Present", bottomLeftCornerOfText, fontScale, fontColor, thickness, lineType)

                if name in students:
                    students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])



    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF ==ord("q"):       # whenever  write q th while loop will break
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

