import numpy as np
import cv2
import os
from fuzzy import *
from model import *
from image import *

def main(image_path, model_path):
    fuzzy_ctrl = setup_fuzzy_logic()
    model = load_model(model_path)
    image = preprocess_image(image_path)
    predictions = run_inference(model, image)
    detected_objects = postprocess_predictions(predictions, threshold)
    
    filename = os.path.basename(image_path)
    file_id = os.path.splitext(filename)[0]
    output_dir = f'output/{file_id}'
    os.makedirs(output_dir, exist_ok=True)

    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    max_confidence = 0 
    max_spill_likelihood = 0  #
    

    for i, (mask, confidence) in enumerate(detected_objects):
        area = np.sum(mask > 0)
        spill_likelihood = calculate_fuzzy_likelihood(area, confidence, fuzzy_ctrl)

        if confidence > max_confidence:
            max_confidence = confidence

        if spill_likelihood > max_spill_likelihood:
            max_spill_likelihood = spill_likelihood

        image_with_current_detection = original_image.copy()

        overlayed_image = overlay_mask_on_image(image_with_current_detection, mask)
        overlay_text_on_image(
            overlayed_image,
            f'Confidence: {confidence*100:.2f}%, Spill Likelihood: {spill_likelihood*100:.2f}%',
            (6, 15)
        )
        print(f'Confidence: {confidence*100:.2f}%, Spill Likelihood: {spill_likelihood*100:.2f}%')

        filepath = f'{output_dir}/detection_{i+1}.png'
        show_and_save_image(overlayed_image, filepath)

    final_image = original_image.copy()
    for mask, _ in detected_objects:
        final_image = overlay_mask_on_image(final_image, mask)

    filepath = f'{output_dir}/all_detections.png'
    overlay_text_on_image(
        final_image,
        f'Max Spill Likelihood: {max_spill_likelihood*100:.2f}%',  
        (6, 15)
    )
    show_and_save_image(final_image, filepath)

    print(f'The max confidence level is {max_confidence*100:.2f}%')
    print(f'The max spill likelihood is {max_spill_likelihood*100:.2f}%')  

if __name__ == "__main__":
    image_path = input("Enter the path to the image: ")
    if image_path == "" : image_path = 'images/6.jpg'
    model_path = mp
    threshold = 0.4
    main(image_path, model_path)
