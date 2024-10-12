import cv2
import numpy as np
from orientation_correction import correct_orientation
from utils import save_from_numpy

# Load the image from file
image_path = 'QCMFINALtilte2.jpg'  
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Image not found at path: {image_path}")

# Correct the orientation of the image
angle, corrected_image = correct_orientation(image)

# Print the angle for debugging purposes
print(f"Detected rotation angle: {angle} degrees")

# Save the corrected image to a file
output_path = r'D:\tryingtilte\InvoiceOrientationCorrection-master\src\corrected\corrected_imagetilte2.jpg'  
save_from_numpy(output_path, corrected_image)

print(f"Corrected image saved to: {output_path}")
