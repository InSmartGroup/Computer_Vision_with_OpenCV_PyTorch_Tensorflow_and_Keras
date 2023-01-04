import cv2
import numpy as np
import convenience_functions as cf

# Read an image
source_image = cv2.imread('..//Data//Images//Trump.jpg', 1)
print(f"Source image shape: {source_image.shape}")
# cf.show_cv2(source_image)

# Convert to grayscale
image_gray = cf.color(source_image, 'bgr', 'gray')

# Create a face classifier
face_classifier = cv2.CascadeClassifier("..//Data//Haarcascades//haarcascade_frontalface_default.xml")

# Detect faces in the image
faces = face_classifier.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=5)

# Check if any faces were detected
if faces is ():
    print('Unable to detect any faces.')

# Process the image
for x, y, w, h in faces:
    cv2.rectangle(source_image, (x, y), (x + w, y + h), (0, 255, 255), 2, lineType=cv2.LINE_8)

cf.show_cv2(source_image)
