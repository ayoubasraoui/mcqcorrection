import cv2
import pytesseract
import numpy as np


# Chargement de l'image du QCM
image = cv2.imread('scan1.png')

# Conversion en niveaux de gris et floutage pour réduire le bruit
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Seuillage pour séparer les zones remplies (réponses) des zones vides
thresholded_image = cv2.threshold(image_blurred, 127, 255, cv2.THRESH_BINARY)[1]

# Détection des contours (cercles) représentant les cases à cocher/bulles de réponse
contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


# Initialisation des listes pour stocker les réponses et les coordonnées des zones de réponse
detected_answers = []
answer_regions = []

# Itération sur chaque contour détecté
for contour in contours:
    # Extraction des coordonnées et de la taille de la zone de réponse
    x, y, w, h = cv2.boundingRect(contour)

    # Extraction de la zone de réponse
    answer_region = image[y:y+h, x:x+w]

    # Application de l'OCR pour extraire le texte de la réponse
    answer_text = pytesseract.image_to_string(answer_region, config='--psm 10')

    # Ajout de la réponse et de sa zone de coordonnées aux listes respectives
    detected_answers.append(answer_text)
    answer_regions.append((x, y, w, h))

    # Dessiner un rectangle bleu autour de la zone de réponse sur l'image d'origine
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Affichage de l'image avec les zones de réponse détectées
cv2.imshow('QCM avec réponses détectées', image)
cv2.waitKey(0)


# Ouverture d'un fichier texte en mode écriture
with open('reponses_qcm.txt', 'w') as f:
    # Écriture de chaque réponse détectée dans le fichier
    for answer in detected_answers:
        f.write(f"{answer}\n")
