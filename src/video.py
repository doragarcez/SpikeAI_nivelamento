import numpy as np
import cv2

def apply_black_background(rgb_image, detection_result):
    if not detection_result.segmentation_masks:
        return rgb_image

    mask = detection_result.segmentation_masks[0].numpy_view()
    mask = np.squeeze(mask)  # remove dimensões de tamanho 1 -> vira (1080, 1920)

    condition = mask > 0.5

    fundo_preto = np.zeros_like(rgb_image)
    output = np.where(condition[..., None], rgb_image, fundo_preto)

    return output.astype(np.uint8)