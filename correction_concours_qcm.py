#__________version1_________________

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
#     return binary, image

# def detect_filled_circles(binary_image):
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     filled_cells = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if 100 < area < 1000:  # Adjust thresholds to ignore large contours like outer circles
#             perimeter = cv2.arcLength(cnt, True)
#             circularity = 4 * np.pi * (area / (perimeter * perimeter))
#             if 0.7 < circularity < 1.2:  # Adjust circularity threshold as needed
#                 (x, y, w, h) = cv2.boundingRect(cnt)
#                 if 10 < w < 50 and 10 < h < 50:  # Filter by size to ignore small and large objects
#                     filled_cells.append((x, y, w, h))

#     return filled_cells

# def classify_filled_cells(filled_cells):
#     filled_cells = sorted(filled_cells, key=lambda b: (b[1], b[0]))  # Sort by y then by x
#     classified_filled_cells = {}
#     for i, (x, y, w, h) in enumerate(filled_cells):
#         question = (i // 3) + 1  # Assuming 3 options per question
#         option = ['A', 'B', 'C'][i % 3]
#         if question not in classified_filled_cells:
#             classified_filled_cells[question] = {}
#         classified_filled_cells[question][option] = (x, y, w, h)
#     return classified_filled_cells

# def format_filled_cells(classified_filled_cells):
#     formatted_cells = []
#     for question, options in classified_filled_cells.items():
#         for option, _ in options.items():
#             formatted_cells.append(f"{question}:{option}")
#     return ', '.join(formatted_cells)

# def main(image_path):
#     binary, image = preprocess_image(image_path)
#     filled_cells = detect_filled_circles(binary)
#     classified_filled_cells = classify_filled_cells(filled_cells)
#     formatted_cells = format_filled_cells(classified_filled_cells)

#     # Plot detected cells
#     for (x, y, w, h) in filled_cells:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title("Detected Answers")
#     plt.show()

#     # Save formatted cells to CSV
#     df = pd.DataFrame([formatted_cells], columns=['Selected Cells'])
#     df.to_csv('detected_cells.csv', index=False)

# image_path = 'D:/Stage Visiativ/qcmpaper.jpg'
# main(image_path)



#_______version2_______________

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
#     return binary, image

# def detect_filled_circles(binary_image):
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filled_cells = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if 100 < area < 1000:  # Adjust thresholds to ignore large contours like outer circles
#             perimeter = cv2.arcLength(cnt, True)
#             circularity = 4 * np.pi * (area / (perimeter * perimeter))
#             if 0.7 < circularity < 1.2:  # Adjust circularity threshold as needed
#                 (x, y, w, h) = cv2.boundingRect(cnt)
#                 # Filter by size and position to ignore small/large objects and exclude corners
#                 if 50 < y < 800 and 50 < x < 550:  
#                     filled_cells.append((x, y, w, h))
#     return filled_cells

# def classify_filled_cells(filled_cells):
#     filled_cells = sorted(filled_cells, key=lambda b: (b[1], b[0]))  # Sort by y then by x
#     classified_filled_cells = {}
#     for i, (x, y, w, h) in enumerate(filled_cells):
#         question = (i // 3) + 1  # Assuming 3 options per question
#         option = ['A', 'B', 'C'][i % 3]
#         if question not in classified_filled_cells:
#             classified_filled_cells[question] = {}
#         classified_filled_cells[question][option] = (x, y, w, h)
#     return classified_filled_cells

# def format_filled_cells(classified_filled_cells):
#     formatted_cells = []
#     for question, options in classified_filled_cells.items():
#         for option, _ in options.items():
#             formatted_cells.append(f"{question}:{option}")
#     return ', '.join(formatted_cells)

# def main(image_path):
#     binary, image = preprocess_image(image_path)
#     filled_cells = detect_filled_circles(binary)
#     classified_filled_cells = classify_filled_cells(filled_cells)
#     formatted_cells = format_filled_cells(classified_filled_cells)

#     # Plot detected cells
#     for (x, y, w, h) in filled_cells:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title("Detected Answers")
#     plt.show()

#     # Save formatted cells to CSV
#     if formatted_cells:
#         cells_dict = {'Selected Cells': [formatted_cells]}
#         df = pd.DataFrame(cells_dict)
#         df.to_csv('detected_cells.csv', index=False)
#         print("CSV file generated successfully with the detected cells.")
#     else:
#         print("No filled cells detected, CSV file not generated.")

# image_path = '/Stage Visiativ/qcmpaper.jpg'
# main(image_path)



#___________version3_____________



# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
#     return binary, image

# def detect_filled_circles(binary_image):
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filled_cells = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if 100 < area < 1000:  # Adjust thresholds to ignore large contours like outer circles
#             perimeter = cv2.arcLength(cnt, True)
#             circularity = 4 * np.pi * (area / (perimeter * perimeter))
#             if 0.7 < circularity < 1.2:  # Adjust circularity threshold as needed
#                 (x, y, w, h) = cv2.boundingRect(cnt)
#                 # Filter by size and position to ignore small/large objects and exclude corners
#                 if 50 < y < 800 and 50 < x < 550:  
#                     filled_cells.append((x, y, w, h))
#     return filled_cells

# def classify_filled_cells(filled_cells):
#     filled_cells = sorted(filled_cells, key=lambda b: (b[1], b[0]))  # Sort by y then by x
#     classified_filled_cells = {}
#     for i, (x, y, w, h) in enumerate(filled_cells):
#         question = (i // 3) + 1  # Assuming 3 options per question
#         option = ['A', 'B', 'C'][i % 3]
#         if question not in classified_filled_cells:
#             classified_filled_cells[question] = {}
#         classified_filled_cells[question][option] = (x, y, w, h)
#     return classified_filled_cells

# def format_filled_cells(classified_filled_cells):
#     formatted_cells = []
#     for question, options in classified_filled_cells.items():
#         for option in ['A', 'B', 'C']:
#             if option in options:
#                 formatted_cells.append(f"{question}:{option}")
#             else:
#                 formatted_cells.append(f"{question}:{option}:")
#     return ', '.join(formatted_cells)

# def main(image_path):
#     binary, image = preprocess_image(image_path)
#     filled_cells = detect_filled_circles(binary)
    
#     # Debug: print detected filled cells
#     print("Detected filled cells (x, y, w, h):", filled_cells)
    
#     classified_filled_cells = classify_filled_cells(filled_cells)
    
#     # Debug: print classified filled cells
#     print("Classified filled cells:", classified_filled_cells)
    
#     formatted_cells = format_filled_cells(classified_filled_cells)
    
#     # Debug: print formatted cells
#     print("Formatted cells:", formatted_cells)

#     # Plot detected cells
#     for (x, y, w, h) in filled_cells:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title("Detected Answers")
#     plt.show()

#     # Save formatted cells to CSV
#     if formatted_cells:
#         cells_dict = {'Selected Cells': [formatted_cells]}
#         df = pd.DataFrame(cells_dict)
#         df.to_csv('detected_filled_cells.csv', index=False)
#         print("CSV file generated successfully with the detected filled cells.")
#     else:
#         print("No filled cells detected, CSV file not generated.")

# image_path = '/Stage Visiativ/qcmpaper.jpg'
# main(image_path)











# _____________version4____________________







# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# def preprocess_image(image_path):
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
#     return binary, image

# def detect_filled_circles(binary_image):
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     filled_cells = []
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if 100 < area < 1000:  # Adjust thresholds to ignore large contours like outer circles
#             perimeter = cv2.arcLength(cnt, True)
#             circularity = 4 * np.pi * (area / (perimeter * perimeter))
#             if 0.7 < circularity < 1.2:  # Adjust circularity threshold as needed
#                 (x, y, w, h) = cv2.boundingRect(cnt)
#                 # Filter by size and position to ignore small/large objects and exclude corners
#                 if 150 < y < 750 and 50 < x < 500:  
#                     filled_cells.append((x, y, w, h))
#     return filled_cells

# def classify_filled_cells(filled_cells):
#     filled_cells = sorted(filled_cells, key=lambda b: (b[1], b[0]))  # Sort by y then by x
#     classified_filled_cells = {}

#     row_heights = np.diff(sorted(set([y for _, y, _, _ in filled_cells])))
#     avg_row_height = np.mean(row_heights) if len(row_heights) > 0 else 1
#     col_widths = np.diff(sorted(set([x for x, _, _, _ in filled_cells])))
#     avg_col_width = np.mean(col_widths) if len(col_widths) > 0 else 1
    
#     for (x, y, w, h) in filled_cells:
#         question = int(round(y / avg_row_height)) + 1
#         option = chr(65 + int(round(x / avg_col_width)))
#         if question not in classified_filled_cells:
#             classified_filled_cells[question] = {}
#         classified_filled_cells[question][option] = (x, y, w, h)
#     return classified_filled_cells

# def format_filled_cells(classified_filled_cells):
#     formatted_cells = []
#     for question, options in classified_filled_cells.items():
#         detected_options = [option for option, coords in options.items() if coords]
#         if detected_options:
#             formatted_cells.append(f"{question}:{detected_options[0]}")  
#     return formatted_cells

# def main(image_path):
#     binary, image = preprocess_image(image_path)
#     filled_cells = detect_filled_circles(binary)
    
#     # Debug: print detected filled cells
#     print("Detected filled cells (x, y, w, h):", filled_cells)
    
#     classified_filled_cells = classify_filled_cells(filled_cells)
    
#     # Debug: print classified filled cells
#     print("Classified filled cells:", classified_filled_cells)
    
#     formatted_cells = format_filled_cells(classified_filled_cells)
    
#     # Debug: print formatted cells
#     print("Formatted cells:", formatted_cells)

#     # Plot detected cells
#     for (x, y, w, h) in filled_cells:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.title("Detected Answers")
#     plt.show()

#     # Save formatted cells to CSV
#     if formatted_cells:
#         cells_dict = {'Selected Cells': formatted_cells}
#         df = pd.DataFrame(cells_dict)
#         df.to_csv('detected_filled_cells.csv', index=False)
#         print("CSV file generated successfully with the detected filled cells.")
#     else:
#         print("No filled cells detected, CSV file not generated.")

# # Replace with the path to your image
# image_path = '/Stage Visiativ/qcmpaper.jpg'
# main(image_path)




#_____________version 5___________________





import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)
    return binary, image

def detect_filled_circles(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_cells = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 100 < area < 1000:  # Adjust thresholds to ignore large contours like outer circles
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            if 0.7 < circularity < 1.2:  # Adjust circularity threshold as needed
                (x, y, w, h) = cv2.boundingRect(cnt)
                # Filter by size and position to ignore small/large objects and exclude corners
                if 50 < y < 800 and 50 < x < 550:  
                    filled_cells.append((x, y, w, h))
    return filled_cells

def classify_filled_cells(filled_cells):
    # Sort by y then by x to maintain the correct order
    filled_cells = sorted(filled_cells, key=lambda b: (b[1], b[0]))  
    
    num_options_per_question = 3
    classified_filled_cells = {}
    question = 1
    current_row_y = filled_cells[0][1]
    current_row = []

    for (x, y, w, h) in filled_cells:
        # Start a new row if y-coordinate changes significantly
        if abs(y - current_row_y) > 10:  # Adjust this threshold as needed
            # Process the current row
            for idx, cell in enumerate(current_row):
                option = ['A', 'B', 'C'][idx % num_options_per_question]
                if question not in classified_filled_cells:
                    classified_filled_cells[question] = {}
                classified_filled_cells[question][option] = cell
                if (idx + 1) % num_options_per_question == 0:
                    question += 1
            # Reset for the next row
            current_row = []
            current_row_y = y
        
        current_row.append((x, y, w, h))
    
    # Process the last row
    for idx, cell in enumerate(current_row):
        option = ['A', 'B', 'C'][idx % num_options_per_question]
        if question not in classified_filled_cells:
            classified_filled_cells[question] = {}
        classified_filled_cells[question][option] = cell
        if (idx + 1) % num_options_per_question == 0:
            question += 1
    
    return classified_filled_cells

def format_filled_cells(classified_filled_cells):
    formatted_cells = []
    for question, options in classified_filled_cells.items():
        selected_option = None
        for option in ['A', 'B', 'C']:
            if option in options:
                selected_option = option
                break
        if selected_option:
            formatted_cells.append(f"{question}:{selected_option}")
        else:
            formatted_cells.append(f"{question}:")
    return ', '.join(formatted_cells)

def main(image_path):
    binary, image = preprocess_image(image_path)
    filled_cells = detect_filled_circles(binary)
    
    # Debug: print detected filled cells
    print("Detected filled cells (x, y, w, h):", filled_cells)
    
    classified_filled_cells = classify_filled_cells(filled_cells)
    
    # Debug: print classified filled cells
    print("Classified filled cells:", classified_filled_cells)
    
    formatted_cells = format_filled_cells(classified_filled_cells)
    
    # Debug: print formatted cells
    print("Formatted cells:", formatted_cells)

    # Plot detected cells
    for (x, y, w, h) in filled_cells:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Answers")
    plt.show()

    # Save formatted cells to CSV
    if formatted_cells:
        cells_dict = {'Selected Cells': [formatted_cells]}
        df = pd.DataFrame(cells_dict)
        df.to_csv('detected_filled_cells.csv', index=False)
        print("CSV file generated successfully with the detected filled cells.")
    else:
        print("No filled cells detected, CSV file not generated.")

image_path = '/Stage Visiativ/QCMFINAL.jpg'
main(image_path)


































