import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import numpy as np
from fer import FER as fer

from PIL import ImageFont, ImageDraw, Image

import face_recognition
import face_registry

emotion_types = ["happy", "sad", "angry", "surprise", "disgust", "fear", "neutral"]
emotion_icons = "😊😟😠😯🤢😱😐"

def start_webcam():
    video_capture = cv2.VideoCapture(0)

    paused = False

    face_locations = []
    face_encodings = []
    face_names = []
    process_index = 0

    emotions = []
    emotion_locations = []

    unknown_index = 0

    detector = fer()
    
    while True:
        ret, frame = video_capture.read()

        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image)

        if not paused:
            if process_index == 0:
                emotions = detector.detect_emotions(frame)

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
                    else:
                        top, right, bottom, left = face_location

                        print("An unknown face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))
                        cropped_image = rgb_small_frame[top:bottom, left:right]

                        cv2.imwrite("./people/unknown" + str(unknown_index) + ".png", cropped_image)
                        unknown_index += 1

                        face_registry.reload_registry()
                    face_names.append(name)

            process_index += 1
            if process_index > 5:
                process_index = 0

        for face in emotions:
            coords = face.get("box")
            ems = face.get("emotions")

            most_certain = {"emotion": None, "strength": 0}
            em_index = 0
            for em_type in emotion_types:
                if ems.get(em_type)> most_certain.get("strength"):
                    most_certain["strength"] = ems.get(em_type)
                    most_certain["emotion"] = emotion_icons[em_index]
                em_index += 1

            text = most_certain["emotion"]

            pointX = int(coords[0] + (coords[2] / 2))
            pointY = int(coords[1] + (coords[3] / 2))

            emotion_locations.append((pointX, pointY, text))

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            emoji = ""

            for (x, y, emotion) in emotion_locations:
                if x > left and x < right and y > top and y < bottom:
                    emoji = emotion


            font = ImageFont.truetype("arial.ttf", size=30)

            text_width, text_height = draw.textsize(name, font=font)
            emoji_width, emoji_height = draw.textsize(emoji, font=font)

            draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))
            draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))

            draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255), font=font)
            draw.text((right - emoji_width - 10, bottom - emoji_height - 5), emoji, fill=(255, 255, 255, 255), font=ImageFont.truetype("noto_emoji.ttf", size=30))

        del draw

        cv2.imshow('Full Webcam Recognition', np.array(image))

        key = cv2.waitKey(1) & 0xFF

        if key == ord('r'):
            print("Reloading registry...")
            face_registry.reload_registry()
            print("Registry reloaded!")
        if key == ord('p'):
            paused = not paused
            print(("Paused" if paused else "Resumed") + " face recognition!")
        if key == ord('q'):
            break


    video_capture.release()
    cv2.destroyAllWindows()