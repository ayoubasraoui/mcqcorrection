import cv2
import numpy as np

def deskew_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None:
        raise ValueError("No lines detected in the image.")

    # Calculate the angle of the lines
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        angles.append(angle)

    # Compute the median angle
    median_angle = np.median(angles)

    # Rotate the image to correct the skew
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Save and show the corrected image
    cv2.imwrite('deskewed_imagejdida4444.jpg', rotated_image)
    cv2.imshow("Deskewed Image", rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
deskew_image('QCMTILTED.jpg')





# import cv2
# import numpy as np
# import pytesseract

# def deskew_image_with_text(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)

#     # Detect text orientation
#     # Perform OCR
#     text_data = pytesseract.image_to_osd(gray, output_type=pytesseract.Output.DICT)
#     rotate_angle = text_data['rotate']
    
#     # Debug: Print the detected rotation angle
#     print(f"Detected rotation angle: {rotate_angle}")

#     # Rotate the image to correct the skew
#     (h, w) = image.shape[:2]
#     center = (w / 2, h / 2)
#     rotation_matrix = cv2.getRotationMatrix2D(center, -rotate_angle, 1.0)  # Rotate by -angle to correct
#     rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

#     # Save and show the corrected image
#     cv2.imwrite('deskewed_image_with_text.jpg', rotated_image)
#     cv2.imshow("Deskewed Image", rotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# # Example usage
# deskew_image_with_text('QCM90.jpg')



