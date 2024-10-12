import cv2

import numpy as np
from typing import Tuple


def correct_orientation(image: np.ndarray) -> Tuple[float, np.ndarray]:
    # Convert the input image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply thresholding to create a binary image with inverted colors
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find the coordinates of non-zero (white) pixels in the binary image
    coords = np.column_stack(np.where(thresh > 0))

    # Calculate the orientation angle of the binary region
    angle = cv2.minAreaRect(coords)[-1]

    # Adjust the angle to be in the range [-30, 30] degrees
    angle = -angle if angle < 30 else 90 - angle

    # Get the height and width of the input image
    h, w = image.shape[:2]

    # Calculate the center of the image
    (c_x, c_y) = (w // 2, h // 2)

    # Compute the rotation matrix for the desired angle
    matrix = cv2.getRotationMatrix2D((c_x, c_y), angle, 1.0)

    # Compute the absolute values of the cosine and sine of the rotation angle
    cos = np.abs(matrix[0, 0])
    sin = np.abs(matrix[0, 1])

    # Calculate the new width and height of the rotated image
    n_w = int((h * sin) + (w * cos))
    n_h = int((h * cos) + (w * sin))

    # Adjust the translation part of the transformation matrix
    matrix[0, 2] += (n_w / 2) - c_x
    matrix[1, 2] += (n_h / 2) - c_y

    # Apply the affine transformation to the input image
    # to correct its orientation
    corrected_image = cv2.warpAffine(image, matrix, (n_w, n_h), borderValue=(255, 255, 255))

    # Return the corrected angle and the corrected image
    return angle, corrected_image




#####################################################working not 100% but does the work########################################################
