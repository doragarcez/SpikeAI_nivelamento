import numpy as np
import cv2

def apply_black_background(rgb_image, detection_result):
    if not detection_result.segmentation_masks:
        return np.zeros_like(rgb_image)

    mask = detection_result.segmentation_mask[0].numpy_view()

    condition = mask > 0.5

    fundo_preto = np.zeros_like(rgb_image)
    output = np.where(condition, rgb_image, fundo_preto)

    return output.astype(np.uint8)