import os

import face_recognition

known_face_encodings = []
known_face_names = []

def register(image, name):
    face_encodings = face_recognition.face_encodings(image)
    if len(face_encodings) > 0:
        face_encoding = face_encodings[0]
        known_face_encodings.append(face_encoding)
        known_face_names.append(name)

def clear_registry():
    known_face_encodings.clear()
    known_face_names.clear()

def reload_registry():
    clear_registry()

    files = os.listdir("./people")

    for i in range(len(files)):
        length = len(files[i])
        register(face_recognition.load_image_file("./people/" + files[i]), files[i][0 : length - 4])