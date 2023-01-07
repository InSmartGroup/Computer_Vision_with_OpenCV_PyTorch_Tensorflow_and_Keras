import cv2
import numpy as np
import matplotlib.pyplot as plt
import convenience_functions as cf

# Read an image
source_image = cv2.imread('..//Data//Images//input.jpg', 1)
print(f"Source image shape: {source_image.shape}")
source_image = cf.resize(source_image, 0.7)
# cf.show_cv2(source_image)

# Flatten the image and calculate histogram using 'plt'
image_flat = source_image.ravel()
# plt.hist(image_flat, bins=256, range=[0, 255])
# plt.show()

# Calculate histograms for each color channel using 'cv2'
color_channels = ('b', 'g', 'r')
for i, color in enumerate(color_channels):
    image_cv2_histogram = cv2.calcHist([source_image], [i], None, [256], [0, 256])
    plt.plot(image_cv2_histogram, c=color)
# plt.show()

# Adaptive Histogram Equalization (CLAHE)
clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(3, 3))  # create a CLAHE object
image_hsv = cf.color(source_image, 'bgr', 'hsv')  # convert an image to HSV
h, s, v = cv2.split(image_hsv)  # split the HSV image
v = clahe.apply(v)  # apply CLAHE to the 'value' channel
image_hsv = cv2.merge((h, s, v))  # merge the image back
image_bgr = cf.color(image_hsv, 'hsv', 'bgr')  # convert the image to BGR
# cf.show_cv2(source_image)
# cf.show_cv2(image_bgr)
