import cv2
import numpy as np
import convenience_functions as cf

# Read an image
image_bgr = cv2.imread('..//Data//Images//opencv_inv.png', 1)
print(f"Source image shape: {image_bgr.shape}")
# cf.show_cv2(image_bgr)

# Convert to grayscale
image_gray = cf.color(image_bgr, 'bgr', 'gray')
print(f"Grayscale image shape: {image_gray.shape}")

# Define several kernels for future morphological operations
kernel_3x3 = np.ones((3, 3))
kernel_5x5 = np.ones((5, 5))
kernel_7x7 = np.ones((7, 7))
kernel_9x9 = np.ones((9, 9))

# Erosion
image_erosion_3x3 = cv2.erode(image_gray, kernel=kernel_3x3, iterations=1)
image_erosion_5x5 = cv2.erode(image_gray,kernel_5x5, iterations=1)
image_erosion_7x7 = cv2.erode(image_gray, kernel_7x7, iterations=1)
image_erosion_9x9 = cv2.erode(image_gray, kernel_9x9, iterations=1)

# images_row_1 = np.hstack([image_erosion_3x3, image_erosion_5x5])
# images_row_2 = np.hstack([image_erosion_7x7, image_erosion_9x9])
# images_stacked = np.vstack([images_row_1, images_row_2])
# cf.show_cv2(images_stacked)

# Dilation
image_dilation_3x3 = cv2.dilate(image_gray, kernel_3x3, iterations=1)
image_dilation_5x5 = cv2.dilate(image_gray, kernel_5x5)
image_dilation_7x7 = cv2.dilate(image_gray, kernel_7x7)
image_dilation_9x9 = cv2.dilate(image_gray, kernel_9x9)

images_row_1 = np.hstack([image_dilation_3x3, image_dilation_5x5])
images_row_2 = np.hstack([image_dilation_7x7, image_dilation_9x9])
images_stacked = np.vstack([images_row_1, images_row_2])
# cf.show_cv2(images_stacked)

# Opening (erosion followed by dilation)
image_opening_3x3 = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel_3x3, iterations=1)
image_opening_5x5 = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel_5x5)
image_opening_7x7 = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel_7x7, iterations=1)
image_opening_9x9 = cv2.morphologyEx(image_gray, cv2.MORPH_OPEN, kernel_9x9)

images_row_1 = np.hstack([image_opening_3x3, image_opening_5x5])
images_row_2 = np.hstack([image_opening_7x7, image_opening_9x9])
images_stacked = np.vstack([images_row_1, images_row_2])
# cf.show_cv2(images_stacked)

# Closing (dilation followed by erosion)
image_closing_3x3 = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel_3x3)
image_closing_5x5 = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel_5x5)
image_closing_7x7 = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel_7x7, iterations=1)
image_closing_9x9 = cv2.morphologyEx(image_gray, cv2.MORPH_CLOSE, kernel_9x9, iterations=1)

images_row_1 = np.hstack([image_closing_3x3, image_closing_5x5])
images_row_2 = np.hstack([image_closing_7x7, image_closing_9x9])
images_stacked = np.vstack([images_row_1, images_row_2])
# cf.show_cv2(images_stacked)

# Playing around with my custom kernels
cf_kernel_cross_5x5 = cf.kernel(size=5, form='cross')
cf_kernel_cross_7x7 = cf.kernel(7, 'cross')
cf_kernel_circle_5x5 = cf.kernel(5, 'circle')
cf_kernel_circle_7x7 = cf.kernel(7, 'circle')

# Erosion using custom kernels
image_erosion_cf_kernel_cross_5x5 = cv2.erode(image_gray, cf_kernel_cross_5x5)
image_erosion_cf_kernel_cross_7x7 = cv2.erode(image_gray, cf_kernel_cross_7x7)
image_erosion_cf_kernel_circle_5x5 = cv2.erode(image_gray, cf_kernel_circle_5x5)
image_erosion_cf_kernel_circle_7x7 = cv2.erode(image_gray, cf_kernel_circle_7x7)

# images_row_1 = np.hstack([image_erosion_cf_kernel_cross_5x5, image_erosion_cf_kernel_cross_7x7])
# images_row_2 = np.hstack([image_erosion_cf_kernel_circle_5x5, image_erosion_cf_kernel_circle_7x7])
# images_stacked = np.vstack([images_row_1, images_row_2])
# cf.show_cv2(images_stacked)

# Dilation using custom kernels
image_dilation_cf_kernel_cross_5x5 = cv2.dilate(image_gray, cf_kernel_cross_5x5)
image_dilation_cf_kernel_cross_7x7 = cv2.dilate(image_gray, cf_kernel_cross_7x7)
image_dilation_cf_kernel_circle_5x5 = cv2.dilate(image_gray, cf_kernel_circle_5x5)
image_dilation_cf_kernel_circle_7x7 = cv2.dilate(image_gray, cf_kernel_circle_7x7)

# images_row_1 = np.hstack([image_dilation_cf_kernel_cross_5x5, image_dilation_cf_kernel_cross_7x7])
# images_row_2 = np.hstack([image_dilation_cf_kernel_circle_5x5, image_dilation_cf_kernel_circle_7x7])
# images_stacked = np.vstack([images_row_1, images_row_2])
# cf.show_cv2(images_stacked)

# Canny edge detection
image_gray = cv2.imread('..//Data//Images//obamafacerecog.jpg', 0)
print(f"Image shape: {image_gray.shape}")
# image_gray = cf.resize(image_gray, 0.35)
image_edge_1 = cv2.Canny(image_gray, threshold1=220, threshold2=230)
image_edge_2 = cv2.Canny(image_gray, 230, 250)
image_edge_3 = cv2.Canny(image_gray, 245, 255)
image_edge_4 = cv2.Canny(image_gray, 150, 180)

images_row_1 = np.hstack([image_edge_1, image_edge_2, image_edge_3, image_edge_4])
cf.show_cv2(images_row_1)
