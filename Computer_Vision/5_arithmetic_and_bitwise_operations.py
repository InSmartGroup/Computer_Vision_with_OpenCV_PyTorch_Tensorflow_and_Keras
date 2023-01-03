import cv2
import numpy as np
import convenience_functions as cf

# Read an image
image_bgr = cv2.imread('..//Data//Images//liberty.jpeg')
print(f"Image shape: {image_bgr.shape}")
image_resized = cf.resize(image_bgr, 0.5)

# Convert to grayscale
image_gray = cf.color(image_resized, 'bgr', 'gray')
# cf.show_cv2(image_gray)

# Create a blank image of the same shape as the source image
image_blank = np.ones_like(image_gray) * 100
print(f"Blank image shape: {image_blank.shape}")
print(f"Blank image max value: {image_blank.max()}")

# Increase the brightness of a grayscale image by adding a value of 100
image_gray_manual = image_gray + image_blank
# cf.show_cv2(image_gray_manual)  # it's clipping

# Increase the brightness of a grayscale image by using the 'cv2.add()' function
image_gray_cv2 = cv2.add(image_gray, image_blank)
# cf.show_cv2(image_gray_cv2)  # it's not clipping

# Decrease the brightness of a grayscale image manually
image_gray_manual = image_gray - image_blank
# cf.show_cv2(image_gray_manual)  # it's clipping

# Decrease the brightness of a grayscale image by using the 'cv2.subtract()' function
image_gray_cv2 = cv2.subtract(image_gray, image_blank)
# cf.show_cv2(image_gray_cv2)  # it's fine

# Increase the contrast of the color image
matrix = np.ones_like(image_bgr) * 1.2
image_contrast_increased = cv2.multiply(np.float64(image_bgr), matrix).astype(np.uint8)
# cf.show_cv2(image_contrast_increased)  # it's clipping

# Avoid clipping
image_contrast_increased = np.clip(cv2.multiply(np.float64(image_bgr), matrix), 0, 255).astype(np.uint8)
# cf.show_cv2(image_contrast_increased)

# Decrease the contrast
matrix = np.ones(image_bgr.shape) * 0.4
image_contrast_decreased = cv2.multiply(np.float64(image_bgr), matrix).astype(np.uint8)
# cf.show_cv2(image_contrast_decreased)

# Create new images
image_rect1 = np.ones((500, 500)).astype(np.uint8) * 255
image_rect1 = cv2.rectangle(image_rect1, (0, 0), (500, 500), (0, 0, 0), 250, lineType=cv2.LINE_8)
image_rect1 = cv2.circle(image_rect1, (int(image_rect1.shape[1] / 2), int(image_rect1.shape[0] / 2)),
                         100, (0, 0, 0), -1, lineType=cv2.LINE_AA)
image_rect2 = np.zeros_like(image_rect1).astype(np.uint8)
image_rect2 = cv2.rectangle(image_rect2, (0, 0), (500, 500), (255, 255, 255), 150, lineType=cv2.LINE_8)
image_rect2 = cv2.circle(image_rect2, (int(image_rect1.shape[1] / 2), int(image_rect1.shape[0] / 2)),
                         100, (0, 0, 0), -1, lineType=cv2.LINE_AA)
print(f"Image 1 shape: {image_rect1.shape}")
print(f"Image 2 shape: {image_rect2.shape}")
# cf.show_cv2(image_rect1)
# cf.show_cv2(image_rect2)

# Apply 'cv2.bitwise_and()' to both images
image_bitwise_and = cv2.bitwise_and(image_rect1, image_rect2)
# cf.show_cv2(image_bitwise_and)

# Apply 'cv2.bitwise_or()'
image_bitwise_or = cv2.bitwise_or(image_rect1, image_rect2)
# cf.show_cv2(image_bitwise_or)

# Apply 'cv2.bitwise_not()'
image_bitwise_not = cv2.bitwise_not(image_rect1, image_rect2)
# cf.show_cv2(image_bitwise_not)

# Apply 'cv2.bitwise_xor()'
image_bitwise_xor = cv2.bitwise_xor(image_rect1, image_rect2)
# cf.show_cv2(image_bitwise_xor)
