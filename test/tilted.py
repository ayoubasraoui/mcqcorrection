# import cv2
# import numpy as np
# import pytesseract

# # If Tesseract is not in your PATH, specify the full path to the executable
# pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract.exe'

# def correct_image_orientation(image_path, output_path):
#     # Load the image
#     image = cv2.imread(image_path)
    
#     # Convert the image to gray scale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # Use pytesseract to detect text and its orientation
#     osd = pytesseract.image_to_osd(gray)
    
#     # Extract the rotation angle from the OSD output
#     rotation_angle = int(osd.split("\nRotate: ")[1].split("\n")[0])
    
#     # Rotate the image to the correct orientation
#     if rotation_angle != 0:
#         (h, w) = image.shape[:2]
#         center = (w // 2, h // 2)
        
#         # Calculate the new bounding dimensions of the rotated image
#         radians = np.deg2rad(rotation_angle)
#         sin = np.abs(np.sin(radians))
#         cos = np.abs(np.cos(radians))
#         new_w = int(h * sin + w * cos)
#         new_h = int(h * cos + w * sin)
        
#         # Adjust the rotation matrix to take into account translation
#         M = cv2.getRotationMatrix2D(center, -rotation_angle, 1.0)
#         M[0, 2] += (new_w - w) / 2
#         M[1, 2] += (new_h - h) / 2
        
#         # Rotate the image with the adjusted matrix and new bounding dimensions
#         rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     else:
#         rotated = image
    
#     # Save the corrected image
#     cv2.imwrite(output_path, rotated)
#     print(f"Saved the corrected image to {output_path}")

# # Example usage
# correct_image_orientation('180.jpg', 'corrected_180.jpg')




import cv2
import numpy as np

def get_median_angle(lines):
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 <= angle <= 45:  # Only consider angles close to vertical
            angles.append(angle)
    if not angles:
        return 0
    median_angle = np.median(angles)
    return median_angle

def get_vertical_lines(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Use Probabilistic Hough Transform to detect lines
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=50, maxLineGap=10)
    if lines is None:
        return []
    return lines

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Calculate new bounding dimensions of the rotated image
    radians = np.deg2rad(angle)
    sin = np.abs(np.sin(radians))
    cos = np.abs(np.cos(radians))
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Adjust the rotation matrix
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Rotate the image
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def correct_image_orientation(image_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Get the vertical lines in the image
    lines = get_vertical_lines(image)
    if lines is None or len(lines) == 0:
        print("No lines detected.")
        return

    # Get the median angle of the vertical lines
    angle = get_median_angle(lines)
    print(f"Detected median angle: {angle}")

    # Rotate the image to make the y-axis parallel to the normal y-axis
    rotated_image = rotate_image(image, angle)
    
    # Save the corrected image
    cv2.imwrite(output_path, rotated_image)
    print(f"Saved the corrected image to {output_path}")

    # Display the final result
    cv2.imshow("Corrected Image", rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
correct_image_orientation('QCMTILTED.jpg', 'corrected_image666.jpg')







