import cv2

import numpy as np


def save_from_numpy(save_path: str, image: np.ndarray):
    cv2.imwrite(save_path, image)
