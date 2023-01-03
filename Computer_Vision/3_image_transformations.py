import cv2
import numpy as np
import convenience_functions as cf

# Create a blank image
# image_blank = np.zeros((600, 600, 3), dtype=np.uint8)
# cf.show_cv2(image_blank)

# Read an image and define its shape
image_bgr = cv2.imread('..//Data//Images//Volleyball.jpeg', 1)
image_bgr = cf.resize(image_bgr, 0.5)  # resized the image for convenience
height, width = image_bgr.shape[0], image_bgr.shape[1]
height_quarter, width_quarter = int(height / 4), int(width / 4)

# Define the translation matrix as a 'float32' data type
matrix = np.array([[1, 0, width_quarter], [0, 1, height_quarter]]).astype(np.float32)
print(f"Translation matrix: {matrix}")

# Apply the matrix to an image
image_translated = cv2.warpAffine(image_bgr, matrix, (width, height))
# cf.show_cv2(image_translated)

# Read and resize an image
image_bgr = cv2.imread('..//Data//Images//Volleyball.jpeg', 1)
image_bgr = cf.resize(image_bgr, 0.5)
height, width = image_bgr.shape[0], image_bgr.shape[1]

# Define the rotation matrix
matrix = cv2.getRotationMatrix2D((image_bgr.shape[1] / 2, image_bgr.shape[0] / 2),
                                 angle=90, scale=0.5)
print(f"Rotation matrix: {matrix}")

# Apply the matrix to the image
image_rotated = cv2.warpAffine(image_bgr, matrix, (width, height))
# cf.show_cv2(image_rotated)

# Horizontal image flipping
image_flipped_np = image_bgr[:, ::-1, :]
# cf.show_cv2(image_flipped_np)
image_flipped_cv2 = cv2.flip(image_bgr, 1)
# cf.show_cv2(image_flipped_cv2)

# Vertical image flipping
image_flipped_np = image_bgr[::-1, :, :]
# cf.show_cv2(image_flipped_np)
image_flipped_cv2 = cv2.flip(image_bgr, 0)
# cf.show_cv2(image_flipped_cv2)

# Image flipping in both directions
image_flipped_np = image_bgr[::-1, ::-1, :]
# cf.show_cv2(image_flipped_np)
image_flipped_cv2 = cv2.flip(image_bgr, -1)
# cf.show_cv2(image_flipped_cv2)
