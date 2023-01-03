import cv2
import numpy as np
import convenience_functions as cf

# Read an image
source_image = cv2.imread('..//Data//Images//chess.JPG', 1)
print(f"Source image shape: {source_image.shape}")
# cf.show_cv2(source_image)

#