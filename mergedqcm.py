import cv2
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import pandas as pd

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width = image.shape[:2]

    # Calculate cropping coordinates (7 cm from top and 2 cm from bottom)
    top_crop = 1200  # approximately 7 cm
    bottom_crop = 236  # approximately 2 cm

    # Crop the image to focus only on the answer section
    cropped_image = image[top_crop:height-bottom_crop, :]

    # Convert to grayscale
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)
    # Dilate edges to close gaps
    dilated_edges = cv2.dilate(edges, None)

    return cropped_image, dilated_edges

def detect_lines(binary_image):
    # Ensure binary_image is single-channel
    if len(binary_image.shape) != 2:
        raise ValueError("Input image for line detection must be single-channel (grayscale).")

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return lines

def draw_lines(image, lines):
    # Ensure image is in BGR format (3 channels)
    if len(image.shape) != 3:
        raise ValueError("Image for drawing lines must be a 3-channel BGR image.")

    # Draw detected lines on the image
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def segment_cells(cropped_image, lines):
    # Create a blank image to mark cells
    cell_image = np.zeros(cropped_image.shape[:2], dtype=np.uint8)  # Use single channel for contours
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(cell_image, (x1, y1), (x2, y2), (255), 2)
    
    # Find contours for cells
    contours, _ = cv2.findContours(cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = [cv2.boundingRect(c) for c in contours]
    cells = sorted(cells, key=lambda b: (b[1], b[0]))  # Sort cells by position
    
    # Estimate cell dimensions
    if cells:
        cell_widths = [w for (x, y, w, h) in cells]
        cell_heights = [h for (x, y, w, h) in cells]
        avg_cell_width = int(np.mean(cell_widths)) if cell_widths else 0
        avg_cell_height = int(np.mean(cell_heights)) if cell_heights else 0
    else:
        avg_cell_width = avg_cell_height = 60  # Default values if no cells are detected
    
    return cells, avg_cell_width, avg_cell_height

def extract_cells(cropped_image, cells):
    cell_images = []
    for (x, y, w, h) in cells:
        cell_image = cropped_image[y:y+h, x:x+w]
        cell_images.append((cell_image, (x, y, w, h)))
    return cell_images

def detect_dots_in_cell(cell_image):
    # Convert to grayscale for dot detection
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use binary thresholding to detect black dots
    _, binary_image = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours for dots in the cell
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours

def mark_dots(image, contours, offset_x, offset_y):
    # Ensure image is in BGR format (3 channels)
    if len(image.shape) != 3:
        raise ValueError("Image for marking dots must be a 3-channel BGR image.")

    # Draw contours on the original image (for visualization purposes)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (offset_x + x, offset_y + y), (offset_x + x + w, offset_y + y + h), (0, 255, 0), 2)
    return image

def analyze_dots(cell_images, cell_width, cell_height, num_questions=70, options=['A', 'B', 'C']):
    answers = {}
    for cell_image, (offset_x, offset_y, w, h) in cell_images:
        dots = detect_dots_in_cell(cell_image)
        for dot in dots:
            x, y, _, _ = cv2.boundingRect(dot)
            # Determine the question number and answer option based on coordinates
            question_number = (offset_y // cell_height) + 1
            option_index = (x // cell_width)
            if question_number <= num_questions and option_index < len(options):
                selected_option = options[option_index]
                if question_number not in answers:
                    answers[question_number] = selected_option
    return answers

def write_results_to_csv(answers, output_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer'])
        for question in range(1, len(answers) + 1):
            answer = answers.get(question, '')
            writer.writerow([question, answer])

def main():
    # Path to the candidate's answer sheet image
    image_path = 'QCMFINAL.jpg'
    try:
        # Preprocess the image
        image, binary_image = preprocess_image(image_path)
        # Detect lines to segment cells
        lines = detect_lines(binary_image)
        # Draw lines for visualization
        image_with_lines = draw_lines(image.copy(), lines)
        cv2.imwrite('lines_detected.jpg', image_with_lines)
        # Segment cells and estimate dimensions
        cells, avg_cell_width, avg_cell_height = segment_cells(binary_image, lines)
        # Extract cells from the table
        cell_images = extract_cells(image, cells)
        # Analyze dots in each cell
        answers = analyze_dots(cell_images, avg_cell_width, avg_cell_height)
        # Write results to a CSV file
        csv_file_path = '/Stage Visiativ/MCQ_answers3.csv'  # Adjusted for demo
        write_results_to_csv(answers, csv_file_path)
        # Mark detected dots on the image for visualization
        for cell_image, (x, y, w, h) in cell_images:
            cell_dots = detect_dots_in_cell(cell_image)
            image = mark_dots(image, cell_dots, x, y)
        # Save the visualized image
        output_image_path = '/Stage Visiativ/MCQ_answers_visualized3.jpg'  # Adjusted for demo
        cv2.imwrite(output_image_path, image)
        print(f"Results written to {csv_file_path}")
        print(f"Visualized image saved to {output_image_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
