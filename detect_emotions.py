import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fer import FER as fer
import numpy as np
from PIL import Image, ImageDraw, ImageFont

emotions_types = ["happy", "sad", "angry", "surprise", "disgust", "fear", "neutral"]

def detect(img):
    detector = fer()
    emotions = detector.detect_emotions(img)

    pil_image = Image.fromarray(img)
    image_draw = ImageDraw.Draw(pil_image)

    for face in emotions:
        coords = face.get("box")
        ems = face.get("emotions")

        image_draw.rectangle(((coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3])), outline=(0, 0, 0, 255))

        most_certain = {"emotion": None, "strength": 0}
        
        for em_type in emotions_types:
            if ems.get(em_type)> most_certain.get("strength"):
                most_certain["strength"] = ems.get(em_type)
                most_certain["emotion"] = em_type

        text = most_certain["emotion"] + ": " + str(int(most_certain["strength"] * 100)) + "%"

        print(text)

        width, height = image_draw.textsize(text, font=ImageFont.truetype("arial.ttf", 20))
        image_draw.rectangle(((coords[0], coords[1] + coords[3] - height - 10), (coords[0] + coords[2], coords[1] + coords[3])), fill=(0, 0, 255), outline=(0, 0, 255))
        image_draw.text((coords[0] + 6, coords[1] + coords[3] - height - 5), text, fill=(255, 255, 255, 255), font=ImageFont.truetype("arial.ttf", 20))


    del image_draw

    pil_image.show()

    return emotions

def detect_top(img):
    detector = fer()
    return detector.top_emotion(img)