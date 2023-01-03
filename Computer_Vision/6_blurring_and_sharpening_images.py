import cv2
import numpy as np
import convenience_functions as cf

# Read an image
image_bgr = cv2.imread('..//Data//Images//flowers.jpeg', 1)
print(f"Image shape: {image_bgr.shape}")
image_bgr_resized = cf.resize(image_bgr, 0.35)
# cf.show_cv2(image_bgr_resized)

# Create a blur kernel
kernel_3x3 = np.ones((3, 3)) / 9
kernel_5x5 = np.ones((5, 5)) / 25
kernel_7x7 = np.ones((7, 7)) / 49

# Apply kernels to the image
image_blurred_3x3 = cv2.filter2D(image_bgr_resized, -1, kernel_3x3)
image_blurred_5x5 = cv2.filter2D(image_bgr_resized, -1, kernel_5x5)
image_blurred_7x7 = cv2.filter2D(image_bgr_resized, -1, kernel_7x7)

image_row_1 = np.hstack([image_bgr_resized, image_blurred_3x3])
image_row_2 = np.hstack([image_blurred_5x5, image_blurred_7x7])
image_stacked = np.vstack([image_row_1, image_row_2])
# cf.show_cv2(image_stacked)

# OpenCV blur
kernel_3x3 = (3, 3)
kernel_5x5 = (5, 5)
kernel_7x7 = (7, 7)

image_blurred_3x3 = cv2.blur(image_bgr_resized, kernel_3x3)
image_blurred_5x5 = cv2.blur(image_bgr_resized, kernel_5x5)
image_blurred_7x7 = cv2.blur(image_bgr_resized, kernel_7x7)

image_row_1 = np.hstack([image_bgr_resized, image_blurred_3x3])
image_row_2 = np.hstack([image_blurred_5x5, image_blurred_7x7])
image_stacked = np.vstack([image_row_1, image_row_2])
# cf.show_cv2(image_stacked)

# Gaussian blur
image_blurred_gaussian_3x3 = cv2.GaussianBlur(image_bgr_resized, (3, 3), 0)
image_blurred_gaussian_5x5 = cv2.GaussianBlur(image_bgr_resized, (5, 5), 0)
image_blurred_gaussian_7x7 = cv2.GaussianBlur(image_bgr_resized, (7, 7), 0)

image_row_1 = np.hstack([image_bgr_resized, image_blurred_gaussian_3x3])
image_row_2 = np.hstack([image_blurred_gaussian_5x5, image_blurred_gaussian_7x7])
image_stacked = np.vstack([image_row_1, image_row_2])
# cf.show_cv2(image_stacked)

# Median blur
image_blurred_median_3x3 = cv2.medianBlur(image_bgr_resized, ksize=3)
image_blurred_median_5x5 = cv2.medianBlur(image_bgr_resized, ksize=5)
image_blurred_median_7x7 = cv2.medianBlur(image_bgr_resized, 7)

image_row_1 = np.hstack([image_bgr_resized, image_blurred_median_3x3])
image_row_2 = np.hstack([image_blurred_median_5x5, image_blurred_median_7x7])
image_stacked = np.vstack([image_row_1, image_row_2])
# cf.show_cv2(image_stacked)
