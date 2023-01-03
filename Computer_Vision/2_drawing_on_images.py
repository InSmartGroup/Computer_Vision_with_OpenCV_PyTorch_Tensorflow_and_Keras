import cv2
import numpy as np
import convenience_functions as cf

# Create a black image
image_black = np.zeros((512, 512, 3), dtype=np.uint8)
print(f"Color image shape: {image_black.shape}")

# Convert the image to grayscale
image_gray = cf.color(image_black, 'bgr', 'gray')
print(f"Grayscale image shape: {image_gray.shape}")
# cf.show_cv2(image_black)

# Drawing a line in the image
image_line = cv2.line(image_black, (0, 0), (512, 512), color=(255, 0, 255), thickness=1, lineType=cv2.LINE_AA)
image_line = cv2.line(image_line, (0, 512), (512, 0), color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
center_y = int(image_line.shape[0] / 2)  # define image width center
center_x = int(image_line.shape[1] / 2)  # define image height center
image_line = cv2.line(image_line, (0, center_y), (center_x * 2, center_y),
                      color=(0, 0, 255), thickness=3, lineType=cv2.LINE_AA)
image_line = cv2.line(image_line, (center_x, 0), (center_x, center_y * 2),
                      color=(255, 0, 0), thickness=4, lineType=cv2.LINE_AA)
# cf.show_cv2(image_line)

# Drawing a rectangle in the image
image_black = np.zeros((512, 512, 3), dtype=np.uint8)
image_rect = cv2.rectangle(image_black, (10, 10), (center_x, center_y),
                           color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)
image_rect = cv2.rectangle(image_rect, (center_x, center_y), (512 - 10, 512 - 10),
                           color=(255, 255, 255), thickness=1, lineType=cv2.LINE_8)
image_rect = cv2.rectangle(image_rect, (10, 512 - 10), (center_x, center_y),
                           color=(0, 0, 255), thickness=3, lineType=cv2.LINE_8)
image_rect = cv2.rectangle(image_rect, (center_x, center_y), (512 - 10, 10),
                           color=(0, 255, 0), thickness=3, lineType=cv2.LINE_8)
# cf.show_cv2(image_rect)

# Drawing circles in the image
image_black = np.zeros((512, 512, 3), dtype=np.uint8)
image_circle = cv2.circle(image_black, (center_x, center_y), int(image_black.shape[0] / 2),
                          color=(0, 255, 255), thickness=4, lineType=cv2.LINE_AA)
image_circle = cv2.circle(image_circle, (center_x, center_y), int(image_black.shape[0] / 3),
                          color=(255, 0, 255), thickness=3, lineType=cv2.LINE_AA)
image_circle = cv2.circle(image_circle, (center_x, center_y), int(image_black.shape[0] / 5),
                          color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
image_circle = cv2.circle(image_circle, (center_x, center_y), int(image_black.shape[0] / 10),
                          color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)  # -1 fills the circle in
# cf.show_cv2(image_circle)

# Drawing polygons in the image
image_black = np.zeros((512, 512, 3), dtype=np.uint8)
points = np.array([[[10, 10], [512 - 10, 512 - 10], [10, 512 - 10]]])  # a triangle
image_poly = cv2.polylines(image_black, points, isClosed=True,
                           color=(255, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# cf.show_cv2(image_poly)

image_black = np.zeros((512, 512, 3), dtype=np.uint8)
points = np.array([[[0, 0], [int(center_x + 75), int(center_y)],
                    [int(center_x), int(center_y + 75)], [512, 512]]])
image_poly = cv2.polylines(image_black, points, isClosed=True,
                           color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
# cf.show_cv2(image_poly)

# Add text to the image
image_black = np.zeros((512, 512, 3), dtype=np.uint8)
text = "Here's some text"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.2
font_color = (0, 255, 0)
image_text = cv2.putText(image_black, text, org=(10, 100), fontFace=font,
                         fontScale=font_scale, color=font_color, thickness=1, lineType=cv2.LINE_AA)
image_text = cv2.putText(image_text, "Here's more text", (10, 275), font, font_scale, color=(255, 255, 0),
                         thickness=1, lineType=cv2.LINE_AA)
image_text = cv2.putText(image_text, "Here's the last line", (10, 450), cv2.FONT_HERSHEY_DUPLEX,
                         font_scale, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
# cf.show_cv2(image_text)
