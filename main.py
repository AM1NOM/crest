import cv2
import pyttsx3
from ultralytics import YOLO



# Open the default camera (0)
cam = cv2.VideoCapture(0)
# Capture one frame
ret, frame = cam.read()
# Release the camera
cam.release()
# Save the image to a file
cv2.imwrite("photo.jpg", frame)
print("Photo saved!")



model = YOLO("yolov8n.pt")  
results = model("photo.jpg")
results[0].show()



#Kamil, write your code here
text = "Hello, I am your Python text to speech AI!"



# Initialize the TTS engine
engine = pyttsx3.init()
# Set properties (optional)
engine.setProperty('rate', 150) # Speed of speech
engine.setProperty('volume', 0.9) # Volume (0.0 to 1.0)
# Convert text to speech
engine.say(text)
# Play the speech
engine.runAndWait()