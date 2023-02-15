import face_registry
import full_webcam

face_registry.reload_registry()

#find_faces.find(face_recognition.load_image_file("unknown.jpg"))

# webcam_recognition.start_webcam()

# print(detect_emotions.detect(cv2.imread("unknown.jpg")))

# webcam_emotions.start_webcam()

full_webcam.start_webcam()