# import cv2
# import numpy as np
# import csv
# import os

# def preprocess_image(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")

#     height, width = image.shape[:2]

#     # Calculate cropping coordinates (7 cm from top and 2 cm from bottom)
#     top_crop = 1200  # approximately 7 cm
#     bottom_crop = 236  # approximately 2 cm

#     # Crop the image to focus only on the answer section
#     cropped_image = image[top_crop:height-bottom_crop, :]

#     # Convert to grayscale
#     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (9, 9), 2)
#     # Thresholding to get a binary image
#     _, binary_image = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

#     # Show the preprocessed image for debugging
#     cv2.imshow("Preprocessed Image", binary_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return cropped_image, binary_image

# def detect_dots(binary_image):
#     # Find contours in the binary image
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def mark_dots(image, contours):
#     # Draw contours on the original image (for visualization purposes)
#     for contour in contours:
#         # Get the bounding box of the contour
#         x, y, w, h = cv2.boundingRect(contour)
#         # Draw a rectangle around each detected dot
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
#     return image

# def analyze_dots(contours, num_questions=70, options=['A', 'B', 'C']):
#     answers = {}
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         # Calculate the approximate question number and option index
#         question_number = (y // 40) + 1  # approximate row number
#         option_index = (x // 60)  # approximate column number
#         if question_number <= num_questions and option_index < len(options):
#             selected_option = options[option_index]
#             if question_number not in answers:
#                 answers[question_number] = selected_option
#     return answers

# def write_results_to_csv(answers, output_file_path):
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
#     with open(output_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Question', 'Answer'])
#         for question in range(1, len(answers) + 1):
#             answer = answers.get(question, '')
#             writer.writerow([question, answer])

# def main():
#     # Path to the candidate's answer sheet image
#     image_path = 'QCMFINAL.jpg'
#     try:
#         # Preprocess the image
#         image, binary_image = preprocess_image(image_path)
#         # Detect dots
#         contours = detect_dots(binary_image)
#         if not contours:
#             print("No dots detected. Please check the image quality and thresholding parameters.")
#             return
#         # Analyze the detected dots and determine selected answers
#         answers = analyze_dots(contours)
#         # Write results to a CSV file
#         csv_file_path = '/Stage Visiativ/MCQ_answers.csv'  # Adjusted for demo
#         write_results_to_csv(answers, csv_file_path)
#         # Mark detected dots on the image for visualization
#         visualized_image = mark_dots(image, contours)
#         # Save the visualized image
#         output_image_path = '/Stage Visiativ/MCQ_answers_visualized.jpg'  # Adjusted for demo
#         cv2.imwrite(output_image_path, visualized_image)
#         print(f"Results written to {csv_file_path}")
#         print(f"Visualized image saved to {output_image_path}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     main()



# import cv2
# import numpy as np
# import csv
# import os

# def preprocess_image(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")

#     height, width = image.shape[:2]

#     # Calculate cropping coordinates (7 cm from top and 2 cm from bottom)
#     top_crop = 1200  # approximately 7 cm
#     bottom_crop = 236  # approximately 2 cm

#     # Crop the image to focus only on the answer section
#     cropped_image = image[top_crop:height-bottom_crop, :]

#     # Convert to grayscale
#     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Use Canny edge detection to find edges
#     edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
#     # Dilate edges to close gaps
#     dilated_edges = cv2.dilate(edges, None)

#     # Show the preprocessed image for debugging
#     cv2.imshow("Preprocessed Image", dilated_edges)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#     return cropped_image, dilated_edges

# def detect_lines(binary_image):
#     # Detect lines using Hough Line Transform
#     lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
#     return lines

# def draw_lines(image, lines):
#     # Draw detected lines on the image
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return image

# def segment_cells(cropped_image, lines):
#     # Create a blank image to mark cells
#     cell_image = np.zeros_like(cropped_image)
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(cell_image, (x1, y1), (x2, y2), (255, 255, 255), 2)
    
#     # Find contours for cells
#     contours, _ = cv2.findContours(cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cells = [cv2.boundingRect(c) for c in contours]
#     return cells

# def extract_cells(cropped_image, cells):
#     cell_images = []
#     for (x, y, w, h) in cells:
#         cell_image = cropped_image[y:y+h, x:x+w]
#         cell_images.append((cell_image, (x, y, w, h)))
#     return cell_images

# def detect_dots_in_cell(cell_image):
#     gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary_image = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
#     # Find contours for dots in the cell
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def mark_dots(image, contours, offset_x, offset_y):
#     # Draw contours on the original image (for visualization purposes)
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image, (offset_x + x, offset_y + y), (offset_x + x + w, offset_y + y + h), (0, 255, 0), 2)
#     return image

# def analyze_dots(cell_images, num_questions=70, options=['A', 'B', 'C']):
#     answers = {}
#     for cell_image, (offset_x, offset_y, w, h) in cell_images:
#         dots = detect_dots_in_cell(cell_image)
#         for dot in dots:
#             x, y, _, _ = cv2.boundingRect(dot)
#             question_number = (offset_y // 40) + 1  # approximate row number
#             option_index = (offset_x // 60)  # approximate column number
#             if question_number <= num_questions and option_index < len(options):
#                 selected_option = options[option_index]
#                 if question_number not in answers:
#                     answers[question_number] = selected_option
#     return answers

# def write_results_to_csv(answers, output_file_path):
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
#     with open(output_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Question', 'Answer'])
#         for question in range(1, len(answers) + 1):
#             answer = answers.get(question, '')
#             writer.writerow([question, answer])

# def main():
#     # Path to the candidate's answer sheet image
#     image_path = 'QCMFINAL.jpg'
#     try:
#         # Preprocess the image
#         image, binary_image = preprocess_image(image_path)
#         # Detect lines to segment cells
#         lines = detect_lines(binary_image)
#         # Draw lines for visualization
#         image_with_lines = draw_lines(image.copy(), lines)
#         cv2.imwrite('lines_detected.jpg', image_with_lines)
#         # Segment cells
#         cells = segment_cells(binary_image, lines)
#         # Extract cells from the table
#         cell_images = extract_cells(image, cells)
#         # Analyze dots in each cell
#         answers = analyze_dots(cell_images)
#         # Write results to a CSV file
#         csv_file_path = '/Stage Visiativ/MCQ_answers4.csv'  # Adjusted for demo
#         write_results_to_csv(answers, csv_file_path)
#         # Mark detected dots on the image for visualization
#         for cell_image, (x, y, w, h) in cell_images:
#             cell_dots = detect_dots_in_cell(cell_image)
#             image = mark_dots(image, cell_dots, x, y)
#         # Save the visualized image
#         output_image_path = '/Stage Visiativ/MCQ_answers_visualized4.jpg'  # Adjusted for demo
#         cv2.imwrite(output_image_path, image)
#         print(f"Results written to {csv_file_path}")
#         print(f"Visualized image saved to {output_image_path}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     main()




# import cv2
# import numpy as np
# import csv
# import os

# def preprocess_image(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")

#     height, width = image.shape[:2]

#     # Calculate cropping coordinates (7 cm from top and 2 cm from bottom)
#     top_crop = 1200  # approximately 7 cm
#     bottom_crop = 236  # approximately 2 cm

#     # Crop the image to focus only on the answer section
#     cropped_image = image[top_crop:height-bottom_crop, :]

#     # Convert to grayscale
#     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Use Canny edge detection to find edges
#     edges = cv2.Canny(blurred, 50, 150)
#     # Dilate edges to close gaps
#     dilated_edges = cv2.dilate(edges, None)

#     return cropped_image, dilated_edges

# def detect_lines(binary_image):
#     # Ensure binary_image is single-channel
#     if len(binary_image.shape) != 2:
#         raise ValueError("Input image for line detection must be single-channel (grayscale).")

#     # Detect lines using Hough Line Transform
#     lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
#     return lines

# def draw_lines(image, lines):
#     # Ensure image is in BGR format (3 channels)
#     if len(image.shape) != 3:
#         raise ValueError("Image for drawing lines must be a 3-channel BGR image.")

#     # Draw detected lines on the image
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return image

# def segment_cells(cropped_image, lines):
#     # Create a blank image to mark cells
#     cell_image = np.zeros(cropped_image.shape[:2], dtype=np.uint8)  # Use single channel for contours
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(cell_image, (x1, y1), (x2, y2), (255), 2)
    
#     # Find contours for cells
#     contours, _ = cv2.findContours(cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cells = [cv2.boundingRect(c) for c in contours]
#     cells = sorted(cells, key=lambda b: (b[1], b[0]))  # Sort cells by position
    
#     # Estimate cell dimensions
#     if cells:
#         cell_widths = [w for (x, y, w, h) in cells]
#         cell_heights = [h for (x, y, w, h) in cells]
#         avg_cell_width = int(np.mean(cell_widths)) if cell_widths else 0
#         avg_cell_height = int(np.mean(cell_heights)) if cell_heights else 0
#     else:
#         avg_cell_width = avg_cell_height = 60  # Default values if no cells are detected
    
#     return cells, avg_cell_width, avg_cell_height

# def extract_cells(cropped_image, cells):
#     cell_images = []
#     for (x, y, w, h) in cells:
#         cell_image = cropped_image[y:y+h, x:x+w]
#         cell_images.append((cell_image, (x, y, w, h)))
#     return cell_images

# def detect_dots_in_cell(cell_image):
#     # Convert to grayscale for dot detection
#     gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Use binary thresholding to detect black dots
#     _, binary_image = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    
#     # Find contours for dots in the cell
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     return contours

# def mark_dots(image, contours, offset_x, offset_y):
#     # Ensure image is in BGR format (3 channels)
#     if len(image.shape) != 3:
#         raise ValueError("Image for marking dots must be a 3-channel BGR image.")

#     # Draw contours on the original image (for visualization purposes)
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image, (offset_x + x, offset_y + y), (offset_x + x + w, offset_y + y + h), (0, 255, 0), 2)
#     return image

# def analyze_dots(cell_images, cell_width, cell_height, num_questions=70, options=['A', 'B', 'C']):
#     answers = {}
#     for cell_image, (offset_x, offset_y, w, h) in cell_images:
#         dots = detect_dots_in_cell(cell_image)
#         for dot in dots:
#             x, y, _, _ = cv2.boundingRect(dot)
#             # Determine the question number and answer option based on coordinates
#             question_number = (offset_y // cell_height) + 1
#             option_index = (x // cell_width)
#             if question_number <= num_questions and option_index < len(options):
#                 selected_option = options[option_index]
#                 if question_number not in answers:
#                     answers[question_number] = selected_option
#     return answers

# def write_results_to_csv(answers, output_file_path):
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
#     with open(output_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Question', 'Answer'])
#         for question in range(1, len(answers) + 1):
#             answer = answers.get(question, '')
#             writer.writerow([question, answer])

# def visualize_detected_dots(image_path, cell_images):
#     image = cv2.imread(image_path)
#     for cell_image, (x, y, w, h) in cell_images:
#         cell_dots = detect_dots_in_cell(cell_image)
#         image = mark_dots(image, cell_dots, x, y)
#     # Display the image with detected dots
#     cv2.imshow("Detected Dots", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def main():
#     # Path to the candidate's answer sheet image
#     image_path = 'QCMFINAL.jpg'
#     try:
#         # Preprocess the image
#         image, binary_image = preprocess_image(image_path)
#         # Detect lines to segment cells
#         lines = detect_lines(binary_image)
#         # Draw lines for visualization
#         image_with_lines = draw_lines(image.copy(), lines)
#         cv2.imwrite('lines_detected1999.jpg', image_with_lines)
        
#         # Now work with the generated 'lines_detected1.jpg' image to detect dots
#         lines_detected_image_path = 'lines_detected1999.jpg'
#         lines_detected_image = cv2.imread(lines_detected_image_path)
#         cells, avg_cell_width, avg_cell_height = segment_cells(binary_image, lines)
#         cell_images = extract_cells(lines_detected_image, cells)
        
#         # Analyze dots in each cell
#         answers = analyze_dots(cell_images, avg_cell_width, avg_cell_height)
#         # Write results to a CSV file
#         csv_file_path = '/Stage Visiativ/MCQ_answers.csv'  # Adjusted for demo
#         write_results_to_csv(answers, csv_file_path)
        
#         # Visualize detected dots on the image
#         visualize_detected_dots(lines_detected_image_path, cell_images)
        
#         print(f"Results written to {csv_file_path}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     main()



################################version final not final xd##############################""


# import cv2
# import numpy as np
# import csv
# import os

# def preprocess_image(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")

#     height, width = image.shape[:2]

#     # Calculate cropping coordinates (7 cm from top and 2 cm from bottom)
#     top_crop = 1200  # approximately 7 cm
#     bottom_crop = 236  # approximately 2 cm

#     # Crop the image to focus only on the answer section
#     cropped_image = image[top_crop:height-bottom_crop, :]

#     # Convert to grayscale
#     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Use Canny edge detection to find edges
#     edges = cv2.Canny(blurred, 50, 150)
#     # Dilate edges to close gaps
#     dilated_edges = cv2.dilate(edges, None)

#     return cropped_image, dilated_edges

# def detect_lines(binary_image):
#     # Ensure binary_image is single-channel
#     if len(binary_image.shape) != 2:
#         raise ValueError("Input image for line detection must be single-channel (grayscale).")

#     # Detect lines using Hough Line Transform
#     lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
#     return lines

# def draw_lines(image, lines):
#     # Ensure image is in BGR format (3 channels)
#     if len(image.shape) != 3:
#         raise ValueError("Image for drawing lines must be a 3-channel BGR image.")

#     # Draw detected lines on the image
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return image

# def segment_cells(cropped_image, lines):
#     # Create a blank image to mark cells
#     cell_image = np.zeros(cropped_image.shape[:2], dtype=np.uint8)  # Use single channel for contours
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(cell_image, (x1, y1), (x2, y2), (255), 2)
    
#     # Find contours for cells
#     contours, _ = cv2.findContours(cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cells = [cv2.boundingRect(c) for c in contours]
#     cells = sorted(cells, key=lambda b: (b[1], b[0]))  # Sort cells by position
    
#     # Estimate cell dimensions
#     if cells:
#         cell_widths = [w for (x, y, w, h) in cells]
#         cell_heights = [h for (x, y, w, h) in cells]
#         avg_cell_width = int(np.mean(cell_widths)) if cell_widths else 0
#         avg_cell_height = int(np.mean(cell_heights)) if cell_heights else 0
#     else:
#         avg_cell_width = avg_cell_height = 60  # Default values if no cells are detected
    
#     return cells, avg_cell_width, avg_cell_height

# def extract_cells(cropped_image, cells):
#     cell_images = []
#     for (x, y, w, h) in cells:
#         cell_image = cropped_image[y:y+h, x:x+w]
#         cell_images.append((cell_image, (x, y, w, h)))
#     return cell_images

# def detect_dots_in_cell(cell_image):
#     # Convert to grayscale for dot detection
#     gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Use binary thresholding to detect black dots
#     _, binary_image = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    
#     # Find contours for dots in the cell
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     return contours

# def mark_dots(image, contours, offset_x, offset_y):
#     # Ensure image is in BGR format (3 channels)
#     if len(image.shape) != 3:
#         raise ValueError("Image for marking dots must be a 3-channel BGR image.")

#     # Draw contours on the original image (for visualization purposes)
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image, (offset_x + x, offset_y + y), (offset_x + x + w, offset_y + y + h), (0, 255, 0), 2)
#     return image

# def analyze_dots(cell_images, cell_width, cell_height, num_questions=70, options=['A', 'B', 'C']):
#     answers = {}
#     for cell_image, (offset_x, offset_y, w, h) in cell_images:
#         dots = detect_dots_in_cell(cell_image)
#         for dot in dots:
#             x, y, _, _ = cv2.boundingRect(dot)
#             # Determine the question number and answer option based on coordinates
#             question_number = (offset_y // cell_height) + 1
#             option_index = (x // cell_width)
#             if question_number <= num_questions and option_index < len(options):
#                 selected_option = options[option_index]
#                 if question_number not in answers:
#                     answers[question_number] = selected_option
#     return answers

# def write_results_to_csv(answers, output_file_path):
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
#     with open(output_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Question', 'Answer'])
#         for question in range(1, len(answers) + 1):
#             answer = answers.get(question, '')
#             writer.writerow([question, answer])

# def visualize_and_save_detected_dots(image_path, cell_images, output_visualized_path):
#     image = cv2.imread(image_path)
#     for cell_image, (x, y, w, h) in cell_images:
#         cell_dots = detect_dots_in_cell(cell_image)
#         image = mark_dots(image, cell_dots, x, y)
#     # Save the image with detected dots
#     cv2.imwrite(output_visualized_path, image)
#     # Display the image with detected dots
#     cv2.imshow("Detected Dots", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def main():
#     # Path to the candidate's answer sheet image
#     image_path = 'QCMFINAL2.jpg'
#     try:
#         # Preprocess the image
#         image, binary_image = preprocess_image(image_path)
#         # Detect lines to segment cells
#         lines = detect_lines(binary_image)
#         # Draw lines for visualization
#         image_with_lines = draw_lines(image.copy(), lines)
#         cv2.imwrite('lines_detectedfinal2.jpg', image_with_lines)
        
#         # Now work with the generated 'lines_detected1.jpg' image to detect dots
#         lines_detected_image_path = 'lines_detectedfinal2.jpg'
#         lines_detected_image = cv2.imread(lines_detected_image_path)
#         cells, avg_cell_width, avg_cell_height = segment_cells(binary_image, lines)
#         cell_images = extract_cells(lines_detected_image, cells)
        
#         # Analyze dots in each cell
#         answers = analyze_dots(cell_images, avg_cell_width, avg_cell_height)
#         # Write results to a CSV file
#         csv_file_path = '/Stage Visiativ/MCQ_answers3.csv'  # Adjusted for demo
#         write_results_to_csv(answers, csv_file_path)
        
#         # Visualize detected dots on the image and save it
#         visualize_and_save_detected_dots(lines_detected_image_path, cell_images, '/Stage Visiativ/MCQ_answers_dots_detectedfinal2.jpg')
        
#         print(f"Results written to {csv_file_path}")
#         print(f"Visualized image with detected dots saved to /Stage Visiativ/MCQ_answers_dots_detected.jpg")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     main()







##################### version tilted image############################



# import cv2
# import numpy as np
# import csv
# import os

# def preprocess_image(image_path):
#     # Load the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")

#     height, width = image.shape[:2]

#     # Calculate cropping coordinates (7 cm from top and 2 cm from bottom)
#     top_crop = 1200  # approximately 7 cm
#     bottom_crop = 236  # approximately 2 cm

#     # Crop the image to focus only on the answer section
#     cropped_image = image[top_crop:height-bottom_crop, :]

#     # Convert to grayscale
#     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Use Canny edge detection to find edges
#     edges = cv2.Canny(blurred, 50, 150)
#     # Dilate edges to close gaps
#     dilated_edges = cv2.dilate(edges, None)

#     return cropped_image, dilated_edges

# def detect_lines(binary_image):
#     # Ensure binary_image is single-channel
#     if len(binary_image.shape) != 2:
#         raise ValueError("Input image for line detection must be single-channel (grayscale).")

#     # Detect lines using Hough Line Transform
#     lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
#     return lines

# def draw_lines(image, lines):
#     # Ensure image is in BGR format (3 channels)
#     if len(image.shape) != 3:
#         raise ValueError("Image for drawing lines must be a 3-channel BGR image.")

#     # Draw detected lines on the image
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return image

# def segment_cells(cropped_image, lines):
#     # Create a blank image to mark cells
#     cell_image = np.zeros(cropped_image.shape[:2], dtype=np.uint8)  # Use single channel for contours
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(cell_image, (x1, y1), (x2, y2), (255), 2)
    
#     # Find contours for cells
#     contours, _ = cv2.findContours(cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cells = [cv2.boundingRect(c) for c in contours]
#     cells = sorted(cells, key=lambda b: (b[1], b[0]))  # Sort cells by position
    
#     # Estimate cell dimensions
#     if cells:
#         cell_widths = [w for (x, y, w, h) in cells]
#         cell_heights = [h for (x, y, w, h) in cells]
#         avg_cell_width = int(np.mean(cell_widths)) if cell_widths else 0
#         avg_cell_height = int(np.mean(cell_heights)) if cell_heights else 0
#     else:
#         avg_cell_width = avg_cell_height = 60  # Default values if no cells are detected
    
#     return cells, avg_cell_width, avg_cell_height

# def extract_cells(cropped_image, cells):
#     cell_images = []
#     for (x, y, w, h) in cells:
#         cell_image = cropped_image[y:y+h, x:x+w]
#         cell_images.append((cell_image, (x, y, w, h)))
#     return cell_images

# def detect_dots_in_cell(cell_image):
#     # Convert to grayscale for dot detection
#     gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
#     # Apply GaussianBlur to reduce noise
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     # Use binary thresholding to detect black dots
#     _, binary_image = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    
#     # Find contours for dots in the cell
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     return contours

# def mark_dots(image, contours, offset_x, offset_y):
#     # Ensure image is in BGR format (3 channels)
#     if len(image.shape) != 3:
#         raise ValueError("Image for marking dots must be a 3-channel BGR image.")

#     # Draw contours on the original image (for visualization purposes)
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image, (offset_x + x, offset_y + y), (offset_x + x + w, offset_y + y + h), (0, 255, 0), 2)
#     return image

# def analyze_dots(cell_images, cell_width, cell_height, num_questions=70, options=['A', 'B', 'C']):
#     answers = {}
#     for cell_image, (offset_x, offset_y, w, h) in cell_images:
#         dots = detect_dots_in_cell(cell_image)
#         for dot in dots:
#             x, y, _, _ = cv2.boundingRect(dot)
#             # Determine the question number and answer option based on coordinates
#             question_number = (offset_y // cell_height) + 1
#             option_index = (x // cell_width)
#             if question_number <= num_questions and option_index < len(options):
#                 selected_option = options[option_index]
#                 if question_number not in answers:
#                     answers[question_number] = selected_option
#     return answers

# def write_results_to_csv(answers, output_file_path):
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
#     with open(output_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Question', 'Answer'])
#         for question in range(1, len(answers) + 1):
#             answer = answers.get(question, '')
#             writer.writerow([question, answer])

# def visualize_and_save_detected_dots(image_path, cell_images, output_visualized_path):
#     image = cv2.imread(image_path)
#     for cell_image, (x, y, w, h) in cell_images:
#         cell_dots = detect_dots_in_cell(cell_image)
#         image = mark_dots(image, cell_dots, x, y)
#     # Save the image with detected dots
#     cv2.imwrite(output_visualized_path, image)
#     # Display the image with detected dots
#     cv2.imshow("Detected Dots", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def correct_image_rotation(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     coords = np.column_stack(np.where(gray > 0))
#     angle = cv2.minAreaRect(coords)[-1]

#     if angle < -45:
#         angle = -(90 + angle)
#     else:
#         angle = -angle

#     center = (image.shape[1] // 2, image.shape[0] // 2)
#     M = cv2.getRotationMatrix2D(center, angle, 1.0)
#     rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#     return rotated

# def main():
#     # Path to the candidate's answer sheet image
#     image_path = 'QCMTILTED.jpg'
#     try:
#         # Preprocess the image
#         image, binary_image = preprocess_image(image_path)
#         # Correct image rotation
#         corrected_image = correct_image_rotation(image)
#         cv2.imwrite('corrected_image2024.jpg', corrected_image)
        
#         # Detect lines to segment cells
#         lines = detect_lines(binary_image)
#         # Draw lines for visualization
#         image_with_lines = draw_lines(corrected_image.copy(), lines)
#         cv2.imwrite('lines_detectedfinal2024.jpg', image_with_lines)
        
#         # Now work with the generated 'lines_detectedfinal2.jpg' image to detect dots
#         lines_detected_image_path = 'lines_detectedfinal2024.jpg'
#         lines_detected_image = cv2.imread(lines_detected_image_path)
#         cells, avg_cell_width, avg_cell_height = segment_cells(binary_image, lines)
#         cell_images = extract_cells(lines_detected_image, cells)
        
#         # Analyze dots in each cell
#         answers = analyze_dots(cell_images, avg_cell_width, avg_cell_height)
#         # Write results to a CSV file
#         csv_file_path = '/Stage Visiativ/MCQ_answers2024.csv'  # Adjusted for demo
#         write_results_to_csv(answers, csv_file_path)
        
#         # Visualize detected dots on the image and save it
#         visualize_and_save_detected_dots(lines_detected_image_path, cell_images, '/Stage Visiativ/MCQ_answers_dots_detectedfinal2024.jpg')
        
#         print(f"Results written to {csv_file_path}")
#         print(f"Visualized image with detected dots saved to /Stage Visiativ/MCQ_answers_dots_detectedfinal2024.jpg")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     main()





############################################################################################



# import cv2
# import numpy as np
# import csv
# import os

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")

#     height, width = image.shape[:2]
#     top_crop = 1130  # approximately 7 cm
#     bottom_crop = 200  # approximately 2 cm

#     cropped_image = image[top_crop:height-bottom_crop, :]
#     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     dilated_edges = cv2.dilate(edges, None)

#     return cropped_image, dilated_edges

# def detect_lines(binary_image):
#     lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
#     return lines

# def draw_lines(image, lines):
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return image

# def segment_cells(cropped_image, lines):
#     cell_image = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(cell_image, (x1, y1), (x2, y2), (255), 2)
    
#     contours, _ = cv2.findContours(cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cells = [cv2.boundingRect(c) for c in contours]
#     cells = sorted(cells, key=lambda b: (b[1], b[0]))

#     if cells:
#         cell_widths = [w for (x, y, w, h) in cells]
#         cell_heights = [h for (x, y, w, h) in cells]
#         avg_cell_width = int(np.mean(cell_widths)) if cell_widths else 0
#         avg_cell_height = int(np.mean(cell_heights)) if cell_heights else 0
#     else:
#         avg_cell_width = avg_cell_height = 60
    
#     return cells, avg_cell_width, avg_cell_height

# def extract_cells(cropped_image, cells):
#     cell_images = []
#     for (x, y, w, h) in cells:
#         cell_image = cropped_image[y:y+h, x:x+w]
#         cell_images.append((cell_image, (x, y, w, h)))
#     return cell_images

# def detect_dots_in_cell(cell_image):
#     gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary_image = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def mark_dots(image, contours, offset_x, offset_y):
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image, (offset_x + x, offset_y + y), (offset_x + x + w, offset_y + y + h), (0, 255, 0), 2)
#     return image

# def analyze_dots(cell_images, cell_width, cell_height, num_questions=70, options=['A', 'B', 'C']):
#     answers = {}
#     for cell_image, (offset_x, offset_y, w, h) in cell_images:
#         dots = detect_dots_in_cell(cell_image)
#         for dot in dots:
#             x, y, _, _ = cv2.boundingRect(dot)
#             question_number = (offset_y // cell_height) + 1
#             option_index = (x // cell_width)
#             if question_number <= num_questions and option_index < len(options):
#                 selected_option = options[option_index]
#                 if question_number not in answers:
#                     answers[question_number] = selected_option
#     return answers

# def write_results_to_csv(answers, output_file_path):
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
#     with open(output_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Question', 'Answer'])
#         for question in range(1, len(answers) + 1):
#             answer = answers.get(question, '')
#             writer.writerow([question, answer])

# def visualize_and_save_detected_dots(image_path, cell_images, output_visualized_path):
#     image = cv2.imread(image_path)
#     for cell_image, (x, y, w, h) in cell_images:
#         cell_dots = detect_dots_in_cell(cell_image)
#         image = mark_dots(image, cell_dots, x, y)
#     cv2.imwrite(output_visualized_path, image)
#     cv2.imshow("Detected Dots", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# def main():
#     image_path = 'QCMTILTED.jpg'
#     try:
#         image, binary_image = preprocess_image(image_path)
#         lines = detect_lines(binary_image)
#         image_with_lines = draw_lines(image.copy(), lines)
#         cv2.imwrite('lines_detectedfinal2.jpg', image_with_lines)
        
#         lines_detected_image_path = 'lines_detectedfinal2.jpg'
#         lines_detected_image = cv2.imread(lines_detected_image_path)
#         cells, avg_cell_width, avg_cell_height = segment_cells(binary_image, lines)
#         cell_images = extract_cells(lines_detected_image, cells)
        
#         answers = analyze_dots(cell_images, avg_cell_width, avg_cell_height)
#         csv_file_path = '/Stage Visiativ/MCQ_answers3.csv'
#         write_results_to_csv(answers, csv_file_path)
        
#         visualize_and_save_detected_dots(lines_detected_image_path, cell_images, '/Stage Visiativ/MCQ_TILTED.jpg')
        
#         print(f"Results written to {csv_file_path}")
#         print(f"Visualized image with detected dots saved to /Stage Visiativ/MCQ_TILTED.jpg")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     main()




#########################################################################################################
# detected text in the terminal




# import cv2
# import numpy as np
# import csv
# import os
# import pytesseract

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")

#     height, width = image.shape[:2]
#     top_crop = 1130  # approximately 7 cm
#     bottom_crop = 200  # approximately 2 cm

#     cropped_image = image[top_crop:height-bottom_crop, :]
#     gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     dilated_edges = cv2.dilate(edges, None)

#     return cropped_image, dilated_edges

# def detect_lines(binary_image):
#     lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
#     return lines

# def draw_lines(image, lines):
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     return image

# def segment_cells(cropped_image, lines):
#     cell_image = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             cv2.line(cell_image, (x1, y1), (x2, y2), (255), 2)
    
#     contours, _ = cv2.findContours(cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cells = [cv2.boundingRect(c) for c in contours]
#     cells = sorted(cells, key=lambda b: (b[1], b[0]))

#     if cells:
#         cell_widths = [w for (x, y, w, h) in cells]
#         cell_heights = [h for (x, y, w, h) in cells]
#         avg_cell_width = int(np.mean(cell_widths)) if cell_widths else 0
#         avg_cell_height = int(np.mean(cell_heights)) if cell_heights else 0
#     else:
#         avg_cell_width = avg_cell_height = 60
    
#     return cells, avg_cell_width, avg_cell_height

# def extract_cells(cropped_image, cells):
#     cell_images = []
#     for (x, y, w, h) in cells:
#         cell_image = cropped_image[y:y+h, x:x+w]
#         cell_images.append((cell_image, (x, y, w, h)))
#     return cell_images

# def detect_dots_in_cell(cell_image):
#     gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary_image = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours

# def read_text_in_rectangle(image, contour):
#     x, y, w, h = cv2.boundingRect(contour)
#     roi = image[y:y+h, x:x+w]
    
#     # Use pytesseract to read the text inside the rectangle
#     text = pytesseract.image_to_string(roi, config='--psm 6')
#     return text.strip()

# def mark_dots(image, contours, offset_x, offset_y):
#     detected_texts = []
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(image, (offset_x + x, offset_y + y), (offset_x + x + w, offset_y + y + h), (0, 255, 0), 2)
        
#         # Read the text inside the rectangle
#         text = read_text_in_rectangle(image, contour)
#         detected_texts.append((text, (offset_x + x, offset_y + y, w, h)))
#     return image, detected_texts

# def analyze_dots(cell_images, cell_width, cell_height, num_questions=70, options=['A', 'B', 'C']):
#     answers = {}
#     all_detected_texts = []
#     for cell_image, (offset_x, offset_y, w, h) in cell_images:
#         dots = detect_dots_in_cell(cell_image)
#         cell_image_with_dots, detected_texts = mark_dots(cell_image, dots, offset_x, offset_y)
#         all_detected_texts.extend(detected_texts)
        
#         for dot in dots:
#             x, y, _, _ = cv2.boundingRect(dot)
#             question_number = (offset_y // cell_height) + 1
#             option_index = (x // cell_width)
#             if question_number <= num_questions and option_index < len(options):
#                 selected_option = options[option_index]
#                 if question_number not in answers:
#                     answers[question_number] = selected_option
#     return answers, all_detected_texts

# def write_results_to_csv(answers, output_file_path):
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
#     with open(output_file_path, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Question', 'Answer'])
#         for question in range(1, len(answers) + 1):
#             answer = answers.get(question, '')
#             writer.writerow([question, answer])

# def visualize_and_save_detected_dots(image_path, cell_images, output_visualized_path):
#     image = cv2.imread(image_path)
#     all_detected_texts = []
#     for cell_image, (x, y, w, h) in cell_images:
#         cell_dots = detect_dots_in_cell(cell_image)
#         image, detected_texts = mark_dots(image, cell_dots, x, y)
#         all_detected_texts.extend(detected_texts)
    
#     cv2.imwrite(output_visualized_path, image)
#     cv2.imshow("Detected Dots", image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     return all_detected_texts

# def main():
#     image_path = 'QCMFINAL.jpg'
#     try:
#         image, binary_image = preprocess_image(image_path)
#         lines = detect_lines(binary_image)
#         image_with_lines = draw_lines(image.copy(), lines)
#         cv2.imwrite('lines_detectedfinal2.jpg', image_with_lines)
        
#         lines_detected_image_path = 'lines_detectedfinal2.jpg'
#         lines_detected_image = cv2.imread(lines_detected_image_path)
#         cells, avg_cell_width, avg_cell_height = segment_cells(binary_image, lines)
#         cell_images = extract_cells(lines_detected_image, cells)
        
#         answers, detected_texts = analyze_dots(cell_images, avg_cell_width, avg_cell_height)
#         csv_file_path = '/Stage Visiativ/correct/MCQ_answersfinaltext.csv'
#         write_results_to_csv(answers, csv_file_path)
        
#         visualize_and_save_detected_dots(lines_detected_image_path, cell_images, '/Stage Visiativ/correct/MCQ_FINAL_with_tilte90.jpg')
        
#         # Print detected texts for review
#         for text, rect in detected_texts:
#             print(f"Detected text: '{text}' in rectangle at {rect}")
        
#         print(f"Results written to {csv_file_path}")
#         print(f"Visualized image with detected dots and texts saved to /Stage Visiativ/correct/MCQ_FINAL_with_tilte90.jpg")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     main()




#####################################################################################




import cv2
import numpy as np
import csv
import os
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width = image.shape[:2]
    top_crop = 1130  # approximately 7 cm
    bottom_crop = 200  # approximately 2 cm

    cropped_image = image[top_crop:height-bottom_crop, :]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    dilated_edges = cv2.dilate(edges, None)

    return cropped_image, dilated_edges

def detect_lines(binary_image):
    lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    return lines

def draw_lines(image, lines):
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def segment_cells(cropped_image, lines):
    cell_image = np.zeros(cropped_image.shape[:2], dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(cell_image, (x1, y1), (x2, y2), (255), 2)
    
    contours, _ = cv2.findContours(cell_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cells = [cv2.boundingRect(c) for c in contours]
    cells = sorted(cells, key=lambda b: (b[1], b[0]))

    if cells:
        cell_widths = [w for (x, y, w, h) in cells]
        cell_heights = [h for (x, y, w, h) in cells]
        avg_cell_width = int(np.mean(cell_widths)) if cell_widths else 0
        avg_cell_height = int(np.mean(cell_heights)) if cell_heights else 0
    else:
        avg_cell_width = avg_cell_height = 60
    
    return cells, avg_cell_width, avg_cell_height

def extract_cells(cropped_image, cells):
    cell_images = []
    for (x, y, w, h) in cells:
        cell_image = cropped_image[y:y+h, x:x+w]
        cell_images.append((cell_image, (x, y, w, h)))
    return cell_images

def detect_dots_in_cell(cell_image):
    gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY_INV)
    
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def mark_dots(image, contours, offset_x, offset_y):
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (offset_x + x, offset_y + y), (offset_x + x + w, offset_y + y + h), (0, 255, 0), 2)
    return image

def detect_text_in_cell(cell_image):
    result = ocr.ocr(cell_image, cls=True)
    if result is None or len(result) == 0:
        return ""
    text = " ".join([line[1][0] for line in result[0]])
    return text

def analyze_dots_and_text(cell_images, cell_width, cell_height, num_questions=70, options=['A', 'B', 'C']):
    answers = {}
    texts = {}
    for cell_image, (offset_x, offset_y, w, h) in cell_images:
        dots = detect_dots_in_cell(cell_image)
        for dot in dots:
            x, y, _, _ = cv2.boundingRect(dot)
            question_number = (offset_y // cell_height) + 1
            option_index = (x // cell_width)
            if question_number <= num_questions and option_index < len(options):
                selected_option = options[option_index]
                if question_number not in answers:
                    answers[question_number] = selected_option
        # Extract text from cell image
        cell_text = detect_text_in_cell(cell_image)
        texts[(offset_x, offset_y, w, h)] = cell_text
    return answers, texts

def write_results_to_csv(answers, texts, output_file_path):
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Question', 'Answer', 'Text'])
        for question in range(1, len(answers) + 1):
            answer = answers.get(question, '')
            text = texts.get((question, 0, 0, 0), '')  # Use a default key for texts
            writer.writerow([question, answer, text])

def visualize_and_save_detected_dots(image_path, cell_images, output_visualized_path):
    image = cv2.imread(image_path)
    for cell_image, (x, y, w, h) in cell_images:
        cell_dots = detect_dots_in_cell(cell_image)
        image = mark_dots(image, cell_dots, x, y)
    cv2.imwrite(output_visualized_path, image)
    cv2.imshow("Detected Dots", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    image_path = 'QCMFINAL.jpg'
    try:
        image, binary_image = preprocess_image(image_path)
        lines = detect_lines(binary_image)
        image_with_lines = draw_lines(image.copy(), lines)
        cv2.imwrite('lines_detectedtilte90.jpg', image_with_lines)
        
        lines_detected_image_path = 'lines_detectedtilte90.jpg'
        lines_detected_image = cv2.imread(lines_detected_image_path)
        cells, avg_cell_width, avg_cell_height = segment_cells(binary_image, lines)
        cell_images = extract_cells(lines_detected_image, cells)
        
        answers, texts = analyze_dots_and_text(cell_images, avg_cell_width, avg_cell_height)
        csv_file_path = '/Stage Visiativ/correct/MCQ_answerstilte90.csv'
        write_results_to_csv(answers, texts, csv_file_path)
        
        visualize_and_save_detected_dots(lines_detected_image_path, cell_images, '/Stage Visiativ/correct/MCQ_tilte90.jpg')
        
        print(f"Results written to {csv_file_path}")
        print(f"Visualized image with detected dots saved to /Stage Visiativ/correct/MCQ_tilte90.jpg")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()





#########################################




# import cv2
# import numpy as np
# import csv
# import os
# from paddleocr import PaddleOCR

# ocr = PaddleOCR(use_angle_cls=True, lang='en')

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"Image not found at {image_path}")
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     edges = cv2.Canny(blurred, 50, 150)
#     return image, edges

# def detect_grid(edges):
#     lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
#     horizontal_lines = []
#     vertical_lines = []
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         if abs(x2 - x1) > abs(y2 - y1):
#             horizontal_lines.append((y1 + y2) // 2)
#         else:
#             vertical_lines.append((x1 + x2) // 2)
#     horizontal_lines = sorted(set(horizontal_lines))
#     vertical_lines = sorted(set(vertical_lines))
#     return horizontal_lines, vertical_lines

# def extract_cells(image, horizontal_lines, vertical_lines):
#     cells = []
#     for i in range(len(horizontal_lines) - 1):
#         for j in range(len(vertical_lines) - 1):
#             top, bottom = horizontal_lines[i], horizontal_lines[i+1]
#             left, right = vertical_lines[j], vertical_lines[j+1]
#             cell = image[top:bottom, left:right]
#             cells.append((cell, (left, top, right-left, bottom-top)))
#     return cells

# def detect_filled_option(cell_image):
#     gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
#     contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if contours:
#         largest_contour = max(contours, key=cv2.contourArea)
#         if cv2.contourArea(largest_contour) > 50:  # Adjust this threshold as needed
#             return True
#     return False

# def detect_text_in_cell(cell_image):
#     try:
#         result = ocr.ocr(cell_image, cls=True)
#         if result and result[0]:
#             return " ".join([line[1][0] for line in result[0]])
#     except Exception as e:
#         print(f"OCR error: {e}")
#     return ""

# def analyze_answer_sheet(cells, num_options=4):
#     answers = {}
#     texts = {}
#     current_question = 0
#     for i, (cell, _) in enumerate(cells):
#         if i % num_options == 0:
#             current_question += 1
#             text = detect_text_in_cell(cell)
#             if text:
#                 texts[current_question] = text
        
#         if detect_filled_option(cell):
#             answers[current_question] = chr(65 + (i % num_options))  # A, B, C, D, ...
    
#     return answers, texts

# def write_results_to_csv(answers, texts, output_file_path):
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
#     with open(output_file_path, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Question', 'Answer', 'Text'])
#         for question in range(1, max(max(answers.keys(), default=0), max(texts.keys(), default=0)) + 1):
#             answer = answers.get(question, '')
#             text = texts.get(question, '')
#             writer.writerow([question, answer, text])

# def main():
#     image_path = 'QCMFINAL.jpg'
#     output_dir = '/Stage Visiativ/'
#     os.makedirs(output_dir, exist_ok=True)
    
#     try:
#         image, edges = preprocess_image(image_path)
#         horizontal_lines, vertical_lines = detect_grid(edges)
#         cells = extract_cells(image, horizontal_lines, vertical_lines)
        
#         answers, texts = analyze_answer_sheet(cells)
        
#         csv_file_path = os.path.join(output_dir, 'MCQ_answersfinalfinal.csv')
#         write_results_to_csv(answers, texts, csv_file_path)
        
#         print(f"Results written to {csv_file_path}")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == '__main__':
#     main()