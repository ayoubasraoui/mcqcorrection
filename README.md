# MCQ Answer Detection Project

## Overview
This project automates the detection of selected answers on multiple-choice question (MCQ) answer sheets. Using image processing techniques, the project identifies marked answers based on a predefined template, facilitating efficient answer recognition for large volumes of answer sheets.

## Features
- **Automated Answer Detection**: Detects selected answers from MCQ answer sheets by analyzing marks within designated answer areas.
- **Template-Based Recognition**: Utilizes a consistent template format for all answer sheets to ensure uniform processing.
- **Image Processing for Precision**: Leverages image processing techniques to identify marker boundaries and detect filled choices accurately.
  
## Project Structure
- **`template_config/`**: Contains configuration files for answer sheet templates, specifying coordinates and layout information.
- **`scripts/`**: Main scripts for running the detection algorithm.
- **`examples/`**: Example images and results showcasing the answer detection process.

## Steps and Result
### Step 1:
Correcting the Orientation of the Answer Sheet
![tilte to good 1](https://github.com/user-attachments/assets/25b4a6db-f7c0-4bad-8624-92424238cac3)

### Step 2:
Cropping the Image so the Script Only Focuses on the Answer Sheet
![good to crop 1](https://github.com/user-attachments/assets/9eaf2e77-4652-429f-86f4-3e0f711e852b)

### Step 3:
Detecting the answers
![crop tp detect 1](https://github.com/user-attachments/assets/be55875a-1b8d-4bc4-b639-fceea94eb277)


