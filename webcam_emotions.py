import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
from fer import FER as fer
import numpy as np


emotions_types = ["happy", "sad", "angry", "surprise", "disgust", "fear", "neutral"]


def start_webcam():
    video_capture = cv2.VideoCapture(0)

    detector = fer()

    paused = False

    emotions = []
    process_index = 0

    unknown_index = 0
    
    while True:
        ret, frame = video_capture.read()

        if process_index == 0 and not paused:
            emotions = detector.detect_emotions(frame)

        process_index += 1
        if process_index > 5:
            process_index = 0

        for face in emotions:
            coords = face.get("box")
            ems = face.get("emotions")

            most_certain = {"emotion": None, "strength": 0}
            
            for em_type in emotions_types:
                if ems.get(em_type)> most_certain.get("strength"):
                    most_certain["strength"] = ems.get(em_type)
                    most_certain["emotion"] = em_type

            text = most_certain["emotion"] + ": " + str(int(most_certain["strength"] * 100)) + "%"

            cv2.rectangle(frame, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 0, 255), 2)
        
            cv2.rectangle(frame, (coords[0], coords[1] + coords[3] - 35), (coords[0] + coords[2], coords[1] + coords[3]), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, text, (coords[0] + 6, coords[1] + coords[3] - 6), font, 1.0, (255, 255, 255), 1)


        cv2.imshow('Webcam Emotions', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            paused = not paused
            print(("Paused" if paused else "Resumed") + " face recognition!")
        if key == ord('q'):
            break


    video_capture.release()
    cv2.destroyAllWindows()