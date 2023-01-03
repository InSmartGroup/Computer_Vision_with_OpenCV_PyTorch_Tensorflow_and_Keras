import cv2
import numpy as np

import convenience_functions as cf

print(f"OpenCV version: {cv2.__version__}")

# Read an image
image_bgr = cv2.imread('..//Data//Images//flowers.jpeg')

# Resize and display an image
image_bgr = cf.resize(image_bgr, 0.7)
# cf.show_cv2(image_bgr)

# Convert to grayscale and show an image
image_gray = cf.color(image_bgr, 'bgr', 'gray')
# cf.show_cv2(image_gray)

# Load another image
image_bgr = cv2.imread('..//Data//Images//castara.jpeg')

# Check image summary
# cf.info(image_bgr)

# Resize the image and display
image_bgr = cf.resize(image_bgr, 0.5)
# cf.show_cv2(image_bgr)

# Convert to grayscale
image_gray = cf.color(image_bgr, 'bgr', 'gray')
# cf.info(image_gray)
# cf.show_cv2(image_gray)

# Split the BGR image into separate color channels
blue, green, red = cv2.split(image_bgr)
# cf.show_cv2(blue)
# cf.show_cv2(green)
# cf.show_cv2(red)

# Create a new blank image of the same shape as the BGR image
image_blank = np.zeros(image_bgr.shape[:2], dtype='uint8')

# Merge the blank image with separated color channels
image_blank_blue = cv2.merge([blue, image_blank, image_blank])
# cf.show_cv2(image_blank_blue)
image_blank_green = cv2.merge([image_blank, green, image_blank])
# cf.show_cv2(image_blank_green)
image_blank_red = cv2.merge([image_blank, image_blank, red])
# cf.show_cv2(image_blank_red)
image_blank_blueRed = cv2.merge([blue, image_blank, red])
# cf.show_cv2(image_blank_blueRed)
image_blank_greenRed = cv2.merge([image_blank, green, red])
# cf.show_cv2(image_blank_greenRed)
image_blank_blueGreen = cv2.merge([blue, green, image_blank])
# cf.show_cv2(image_blank_blueGreen)
image_blank_fullcolor = cv2.merge([blue, green, red])
# cf.show_cv2(image_blank_fullcolor)

# Convert the image to HSV
image_hsv = cf.color(image_bgr, 'bgr', 'hsv')
# cf.show_cv2(image_hsv)

# Split the HSV image to separate color channels
hue, saturation, value = cv2.split(image_hsv)
# cf.show_cv2(hue)  # the color itself
# cf.show_cv2(saturation)  # color intensity
# cf.show_cv2(value)  # the amount of light

# Playing around with HSV color channels
hue = hue + 20
saturation = saturation + 30
value = np.clip(value + 10, 0, 255)
print(value.max())
image_hsv_merged = cv2.merge([hue, saturation, value])
image_hsv_merged = cf.color(image_hsv_merged, 'hsv', 'bgr')
cf.show_cv2(image_hsv_merged)
