import cv2
import matplotlib.pyplot as plt
import numpy as np


def info(image):
    """Prints out a short summary about the image"""
    print(f"Shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Image size: {image.size} pixels")


def show_mat(image, name='Image', axes=None, grid=None):
    """Displays a BGR image using Matplotlib"""
    plt.title(name)
    plt.imshow(image[:, :, ::-1])
    if axes == 'off':
        plt.axis('off')
    if grid == 'off':
        plt.grid()
    plt.show()


def show_cv2(image, name='Image'):
    """Displays a BGR image using OpenCV"""
    cv2.imshow(name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def resize(image, scale=None):
    """Resizes an image"""
    if scale is not None:
        image = cv2.resize(image, None, fx=scale, fy=scale)

    return image


def color(image, in_color=None, out_color=None):
    """Converts an image to a specified color scheme.
    Supported color schemes: 'bgr', 'rgb', 'gray', 'hsv'"""
    if in_color == 'bgr' and out_color == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if in_color == 'rgb' and out_color == 'bgr':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if in_color == 'gray' and out_color == 'bgr':
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if in_color == 'gray' and out_color == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if in_color == 'gray' and out_color == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if in_color == 'rgb' and out_color == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if in_color == 'bgr' and out_color == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if in_color == 'rgb' and out_color == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    if in_color == 'bgr' and out_color == 'hsv':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if in_color == 'hsv' and out_color == 'bgr':
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    if in_color == 'hsv' and out_color == 'rgb':
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    if in_color == 'hsv' and out_color == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    return image


def matrix_translation(x=None, y=None):
    """Returns a translation matrix"""
    matrix = np.array([[1, 0, x], [0, 1, y]]).astype(np.float32)

    return matrix


def matrix_rotation(x=None, y=None, angle=None, scale=1):
    """Returns a rotation matrix"""
    matrix = cv2.getRotationMatrix2D((x, y), angle, scale)

    return matrix


def kernel(size=3, form='square'):
    """Returns a kernel of specified shape"""
    try:
        if size == 3 and form == 'square':
            k = np.ones((3, 3), dtype=np.uint8)
        elif size == 3 and form == 'cross':
            k = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        elif size == 3 and form == 'circle':
            k = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        if size == 5 and form == 'square':
            k = np.ones((5, 5), dtype=np.uint8)
        elif size == 5 and form == 'cross':
            k = np.array([[0, 0, 1, 0, 0],
                          [0, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 0],
                          [0, 0, 1, 0, 0]], dtype=np.uint8)
        elif size == 5 and form == 'circle':
            k = np.array([[0, 0, 1, 0, 0],
                          [0, 1, 0, 1, 0],
                          [1, 0, 0, 0, 1],
                          [0, 1, 0, 1, 0],
                          [0, 0, 1, 0, 0]], dtype=np.uint8)
        if size == 7 and form == 'square':
            k = np.ones((7, 7), dtype=np.uint8)
        elif size == 7 and form == 'cross':
            k = np.array([[0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 1, 1, 0, 0],
                          [0, 1, 1, 1, 1, 1, 0],
                          [1, 1, 1, 1, 1, 1, 1],
                          [0, 1, 1, 1, 1, 1, 0],
                          [0, 0, 1, 1, 1, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
        elif size == 7 and form == 'circle':
            k = np.array([[0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 1, 0, 0],
                          [0, 1, 0, 0, 0, 1, 0],
                          [1, 0, 0, 0, 0, 0, 1],
                          [0, 1, 0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0, 0, 0]], dtype=np.uint8)
        if size == 9 and form == 'square':
            k = np.ones((9, 9), dtype=np.uint8)
        elif size == 9 and form == 'cross':
            k = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 0, 0, 0],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1, 1, 1, 1, 1],
                          [0, 0, 0, 1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=np.uint8)
        elif size == 9 and form == 'circle':
            k = np.array([[0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0, 1, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 1, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 1, 0],
                          [1, 0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 1, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 1, 0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 1, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0, 0, 0]], dtype=np.uint8)
    except Exception as error:
        print(f"{error=}")
        print("Supported kernel sizes are 3, 5, 7, and 9")
        print("Supported kernel forms are: square, cross, and circle")

    return k


def contours_centroid(contours, image=None, mark=None):
    """Returns centers of mass for each contour in the input list"""
    centers = []
    for contour in contours:
        m = cv2.moments(contour)
        x = int(m['m10'] / m['m00'])
        y = int(m['m01'] / m['m00'])
        centers.append([x, y])
        if image is not None and mark:
            cv2.circle(image, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)

    return centers


def contours_area(contours, filter_area=None, filter_perimeter=None,
                  sort_area=False, sort_perimeter=False, verbose=False):
    """Returns two lists that contain area and perimeter of each contour.
    Syntax:
    contours: var - required variable that contains a list of contours
    filter_area: int - returns only areas that are larger than the specified value
    filter_perimeter: int - returns only perimeters that are larger than the specified value
    sort_area: bool - returns a sorted list of contour areas
    sort_perimeter: bool - returns a sorted list of contour perimeters
    verbose: bool - prints a summary on each contour"""
    areas, perimeters = [], []
    for index, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, closed=True)

        if filter_area is not None:
            if area >= filter_area:
                areas.append(round(area, 3))
        else:
            areas.append(round(area, 3))

        if filter_perimeter is not None:
            if perimeter >= filter_perimeter:
                perimeters.append(round(perimeter, 3))
        else:
            perimeters.append(round(perimeter, 3))

        if verbose:
            print(f"Contour {index} has area: {round(area, 3)} and perimeter: {round(perimeter, 3)}")

    if sort_area:
        areas = sorted(areas, reverse=True)
    if sort_perimeter:
        perimeters = sorted(perimeters, reverse=True)

    return areas, perimeters
