import cv2
import numpy as np
from skimage.filters.thresholding import threshold_local
import convenience_functions as cf

# Read an image
image_bgr = cv2.imread('..//Data//Images//scan.jpg', 1)
print(f"Image shape: {image_bgr.shape}")

# Convert to grayscale
image_gray = cf.color(image_bgr, 'bgr', 'gray')
# cf.show_cv2(image_gray)

# Image thresholding
retval, image_thresh_binary = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY)
retval, image_thresh_binary_inv = cv2.threshold(image_gray, 127, 255, cv2.THRESH_BINARY_INV)
retval, image_thresh_trunc = cv2.threshold(image_gray, 127, 255, cv2.THRESH_TRUNC)
retval, image_thresh_tozero = cv2.threshold(image_gray, 127, 255, cv2.THRESH_TOZERO)
retval, image_thresh_tozero_inv = cv2.threshold(image_gray, 127, 255, cv2.THRESH_TOZERO_INV)
retval, image_thresh_otsu = cv2.threshold(image_gray, thresh=127, maxval=255, type=cv2.THRESH_OTSU)

# cf.show_cv2(image_thresh_binary)
# cf.show_cv2(image_thresh_binary_inv)
# cf.show_cv2(image_thresh_trunc)
# cf.show_cv2(image_thresh_tozero)
# cf.show_cv2(image_thresh_tozero_inv)
# cf.show_cv2(image_thresh_otsu)

# Adaptive thresholding
image_thresh_adaptive_no_blur = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                              cv2.THRESH_BINARY, blockSize=7, C=21)

# Apply gaussian blur prior to using adaptive threshold
image_thresh_adaptive_blur = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 7, 21)
image_row = np.hstack([image_thresh_adaptive_no_blur, image_thresh_adaptive_blur])
cf.show_cv2(image_row)

# Thresholding using scikit-image
hue, saturation, value = cv2.split(image_bgr)
threshold = threshold_local(value, 25, offset=15, method='gaussian')
image_thresh_skimage = (value > threshold).astype(np.uint8) * 255
# cf.show_cv2(image_thresh_skimage)
