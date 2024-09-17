import matplotlib.pyplot as plt
import numpy as np
import cv2

def postprocess_predictions(predictions, threshold=0.4):
    masks = predictions[0]['masks']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']

    detected_objects = []
    for i in range(len(scores)):
        if scores[i] > threshold:
            mask = masks[i, 0].cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8)
            detected_objects.append((mask, scores[i].item()))
    
    return detected_objects


def overlay_mask_on_image(image, mask):
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = (0, 255, 0)
    overlayed_image = cv2.addWeighted(image, 1, color_mask, 0.5, 0)
    return overlayed_image

def overlay_text_on_image(image, text, position, font_scale=0.6, color=(255, 0, 0), thickness=2):
    cv2.putText(image, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def show_and_save_image(image, filepath):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(filepath)
    plt.show()        