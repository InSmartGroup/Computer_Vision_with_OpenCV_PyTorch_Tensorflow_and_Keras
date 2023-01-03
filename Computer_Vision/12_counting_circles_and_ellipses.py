import cv2
import numpy as np
import convenience_functions as cf

# Read an image
source_image = cv2.imread('..//Data//Images//blobs.jpg', 1)
# print(f"Source image shape: {source_image.shape}")
# cf.show_cv2(source_image)

image_gray = cf.color(source_image, 'bgr', 'gray')

val, image_thresh = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY)

blob_detector = cv2.SimpleBlobDetector_create()

blobs = blob_detector.detect(image_thresh)
print(len(blobs))

blank = np.zeros((1, 1))
image_blobs = cv2.drawKeypoints(source_image, blobs, blank, (255, 0, 0))
cf.show_cv2(image_blobs)