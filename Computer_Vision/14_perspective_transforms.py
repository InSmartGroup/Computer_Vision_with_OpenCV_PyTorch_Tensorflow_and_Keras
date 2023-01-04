import cv2
import numpy as np
import convenience_functions as cf

# Read an image
source_image = cv2.imread("..//Data//Images//scan.jpg", 1)
print(f"Source image shape: {source_image.shape}")

# Convert to grayscale
image_gray = cf.color(source_image, 'bgr', 'gray')

# Threshold (comment out one of the following threshold versions)
# image_thresh = cv2.adaptiveThreshold(image_gray, 255, cv2.CALIB_CB_ADAPTIVE_THRESH,
#                                      cv2.THRESH_BINARY, 17, 37)  # adaptive threshold
image_thresh = cv2.inRange(image_gray, 180, 255)  # binary threshold
# cf.show_cv2(image_thresh)

# Find contours
contours, hierarchy = cv2.findContours(image_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Calculate contours area and perimeter
areas, perimeters = cf.contours_area_perimeter(contours, verbose=True)

# Draw contours to see the one we need
# image_contours = cv2.drawContours(source_image, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)
# cf.show_cv2(image_contours)

# Convert the contours tuple to a list and leave only the largest one
contours = list(contours)
contour = contours.pop(4)

# Retrieve the needed 4 points of the largest contour
contour_points = cv2.approxPolyDP(contour, epsilon=cv2.arcLength(contour, True) * 0.1, closed=True)
print(f"Contour points found: {len(contour_points)}")

# Define the size of a transformed image (the document only)
size = (500, 800)

# Define input and output points to perform perspective transform
input_points = np.float32(contour_points)
output_points = np.float32([[0, 0],
                            [0, 800],
                            [500, 800],
                            [500, 0]])

# Calculate the transformation matrix
matrix = cv2.getPerspectiveTransform(input_points, output_points)

# Apply the transformation matrix to the image using the 'cv2.warpPerspective()'
image_transformed = cv2.warpPerspective(source_image, matrix, size)

# Display the input and transformed images
cv2.imshow("Input image", source_image)
cv2.imshow("Transformed image", image_transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
