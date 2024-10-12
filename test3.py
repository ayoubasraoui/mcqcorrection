import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Detect contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on their area (we assume markers are the largest contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the four markers
    markers = contours[:4]

    # Get the coordinates of the markers
    marker_coords = []
    for marker in markers:
        x, y, w, h = cv2.boundingRect(marker)
        marker_coords.append((x, y))

    # Order the markers (top-left, top-right, bottom-right, bottom-left)
    marker_coords = sorted(marker_coords, key=lambda k: [k[1], k[0]])

    # Correct the perspective of the image
    pts1 = np.float32(marker_coords)
    pts2 = np.float32([[0, 0], [800, 0], [800, 1000], [0, 1000]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(image, matrix, (800, 1000))

    return warped

def detect_answers(warped_image, num_questions=70, num_choices=3, num_columns=4):
    questions_per_column = [20, 20, 20, 10]
    total_height = warped_image.shape[0]
    total_width = warped_image.shape[1]
    
    roi_height = total_height // 20  # Height per question (assuming 20 questions per column)
    roi_width = total_width // (num_choices * num_columns)  # Width per choice

    answers = []

    question_number = 1
    for col in range(num_columns):
        for i in range(questions_per_column[col]):
            max_non_zero_count = 0
            selected_choice = None
            for j in range(num_choices):
                x = col * num_choices * roi_width + j * roi_width
                y = i * roi_height
                roi = warped_image[y:y+roi_height, x:x+roi_width]

                # Convert ROI to grayscale and apply thresholding
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, thresh_roi = cv2.threshold(gray_roi, 150, 255, cv2.THRESH_BINARY_INV)

                # Count the number of non-zero pixels
                non_zero_count = cv2.countNonZero(thresh_roi)

                # Determine the most marked choice for the question
                if non_zero_count > max_non_zero_count:
                    max_non_zero_count = non_zero_count
                    selected_choice = chr(65 + j)  # A, B, C, etc.

            # Add the most marked choice to the answers list
            if selected_choice is not None:
                answers.append((question_number, selected_choice))

            question_number += 1

    return answers

def write_answers_to_file(answers, output_file):
    with open(output_file, 'w') as file:
        for question, answer in answers:
            file.write(f"Question {question}: {answer}\n")

def main(image_path, output_file):
    warped_image = preprocess_image(image_path)
    detected_answers = detect_answers(warped_image)
    write_answers_to_file(detected_answers, output_file)
    print("Detected answers written to", output_file)

# Path to the uploaded image
image_path = 'scan1.jpg'
output_file = 'detected_answers_final_corrected.txt'
main(image_path, output_file)

