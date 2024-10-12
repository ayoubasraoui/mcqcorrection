
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def splitBoxes(img, num_rows, num_cols):
    img_height, img_width = img.shape[:2]
    row_height = img_height // num_rows
    col_width = img_width // num_cols
    boxes = []

    for r in range(num_rows):
        for c in range(num_cols):
            start_y = r * row_height
            start_x = c * col_width
            end_y = (r + 1) * row_height if r < num_rows - 1 else img_height
            end_x = (c + 1) * col_width if c < num_cols - 1 else img_width
            box = img[start_y:end_y, start_x:end_x]
            boxes.append(box)
    return boxes

def detectFilledCells(boxes, threshold=200):
    filled_cells = []
    for idx, box in enumerate(boxes):
        mean_intensity = cv2.mean(cv2.cvtColor(box, cv2.COLOR_BGR2GRAY))[0]
        if mean_intensity < threshold:  # Lower mean intensity indicates a filled cell
            filled_cells.append(idx)
    return filled_cells

def saveDetectedAnswersToCSV(detected_answers, filename='detected_answers.csv'):
    df = pd.DataFrame(detected_answers, columns=['Question', 'Choice'])
    df.to_csv(filename, index=False)

# Load the image
image_path = '/Stage Visiativ/test 2/mcq_sheet.jpg'  # Update to the correct path if needed
if not os.path.exists(image_path):
    raise FileNotFoundError(f"The file {image_path} does not exist. Please check the file path.")

img = cv2.imread(image_path)
if img is None:
    raise IOError(f"Unable to read the image file {image_path}. Please check the file.")

# Preprocess the image
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)

# Apply adaptive thresholding
imgThresh = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

# Apply morphological operations to clean the image
kernel = np.ones((5, 5), np.uint8)
imgMorph = cv2.morphologyEx(imgThresh, cv2.MORPH_CLOSE, kernel)

# Find contours
contours, hierarchy = cv2.findContours(imgMorph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours to only those in the answer region (height from 300 to 800)
filtered_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    if 300 < y < 800:
        filtered_contours.append(contour)

# Visualize filtered contours
imgContours = img.copy()
cv2.drawContours(imgContours, filtered_contours, -1, (0, 255, 0), 3)
cv2.imwrite('filtered_contours.jpg', imgContours)

# Total number of questions and choices
total_questions = 70
num_choices = 3

# List to store all detected answers
all_detected_answers = []

# Process each detected contour
for contour_index, contour in enumerate(filtered_contours):
    x, y, w, h = cv2.boundingRect(contour)
    imgCropped = img[y:y+h, x:x+w]
    
    # Convert to grayscale
    imgFilteredGray = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, imgFilteredThresh = cv2.threshold(imgFilteredGray, 150, 255, cv2.THRESH_BINARY_INV)

    # Estimate number of rows based on the height of the contour
    # Each contour has 4 columns
    num_rows = h // (total_questions // len(filtered_contours))

    # Split the image into boxes
    boxes = splitBoxes(imgCropped, num_rows, num_choices)

    # Detect filled cells
    filled_cells = detectFilledCells(boxes, threshold=200)

    # Adjust question numbers based on contour position
    detected_answers = []
    for idx in filled_cells:
        question = idx // num_choices + 1 + (contour_index * num_rows)  # Question number (1-based index)
        choice = chr((idx % num_choices) + 65)  # Choice (A=65, B=66, C=67, etc.)
        detected_answers.append([question, choice])

    all_detected_answers.extend(detected_answers)

# Save all detected answers to CSV
saveDetectedAnswersToCSV(all_detected_answers)
print(f"Detected answers saved to 'detected_answers.csv'.")

# Visualize the results using matplotlib
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# Filtered Contours Image
ax[0].imshow(cv2.cvtColor(imgContours, cv2.COLOR_BGR2RGB))
ax[0].set_title('Filtered Contours Image')

# Filtered Image with Filled Cells
imgFilledCells = imgContours.copy()
for contour in filtered_contours:
    x, y, w, h = cv2.boundingRect(contour)
    imgCropped = img[y:y+h, x:x+w]
    num_cols = num_choices
    num_rows = h // (total_questions // len(filtered_contours))
    boxes = splitBoxes(imgCropped, num_rows, num_cols)
    filled_cells = detectFilledCells(boxes, threshold=200)
    row_height = imgCropped.shape[0] // num_rows
    col_width = imgCropped.shape[1] // num_cols
    for idx in filled_cells:
        row = idx // num_cols
        col = idx % num_cols
        start_y = y + row * row_height
        start_x = x + col * col_width
        end_y = y + (row + 1) * row_height if row < num_rows - 1 else y + h
        end_x = x + (col + 1) * col_width if col < num_cols - 1 else x + w
        cv2.rectangle(imgFilledCells, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)

ax[1].imshow(cv2.cvtColor(imgFilledCells, cv2.COLOR_BGR2RGB))
ax[1].set_title('Detected Filled Cells')

plt.tight_layout()
plt.savefig('detection_visualization_final.png')
plt.show()


