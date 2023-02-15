import cv2
import numpy as np
import tkinter

import face_recognition
import face_registry


def start_webcam():
    video_capture = cv2.VideoCapture(0)

    paused = False

    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    unknown_index = 0

    loader_window = tkinter.Tk()
    loader_window.title("Load faces")
    loader_window.configure(width=512, height=512)
    
    while True:
        ret, frame = video_capture.read()

        if not paused:
            if process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding, face_location in zip(face_encodings, face_locations):
                    matches = face_recognition.compare_faces(face_registry.known_face_encodings, face_encoding)
                    name = "Unknown"
                    
                    face_distances = face_recognition.face_distance(face_registry.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = face_registry.known_face_names[best_match_index]
                        print("Found face of " + name + " on image")
                    else:
                        top, right, bottom, left = face_location

                        print("An unknown face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                        cropped_image = rgb_small_frame[top:bottom, left:right]

                        cv2.imwrite("./people/unknown" + str(unknown_index) + ".png", cropped_image)
                        unknown_index += 1

                        face_registry.reload_registry()
                    face_names.append(name)

            process_this_frame = not process_this_frame
            
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                print(face_names)

        cv2.imshow('Webcam Recognition', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            print("Reloading registry...")
            face_registry.reload_registry()
            print("Registry reloaded!")
        if key == ord('l'):
            loader_window.mainloop()
        if key == ord('p'):
            paused = not paused
            print(("Paused" if paused else "Resumed") + " face recognition!")
        if key == ord('q'):
            break


    video_capture.release()
    cv2.destroyAllWindows()