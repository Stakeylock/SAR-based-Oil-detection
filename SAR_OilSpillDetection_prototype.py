import torch
import torchvision.models.detection as detection
import torchvision.transforms as T
import numpy as np
import cv2
import os
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt

mp = 'model/SAR_model.pth'

def setup_fuzzy_logic():
    area = ctrl.Antecedent(np.arange(0, 10000, 1), 'area')
    confidence = ctrl.Antecedent(np.arange(0, 1, 0.01), 'confidence')
    spill_likelihood = ctrl.Consequent(np.arange(0, 1, 0.01), 'spill_likelihood')

    area['small'] = mf.trapmf(area.universe, [0, 0, 500, 1000])
    area['medium'] = mf.trimf(area.universe, [500, 1500, 3000])
    area['large'] = mf.trapmf(area.universe, [2000, 3000, 5000, 10000])

    confidence['low'] = mf.trapmf(confidence.universe, [0, 0, 0.3, 0.5])
    confidence['medium'] = mf.trimf(confidence.universe, [0.3, 0.5, 0.7])
    confidence['high'] = mf.trapmf(confidence.universe, [0.5, 0.7, 0.9, 1])

    spill_likelihood['unlikely'] = mf.trimf(spill_likelihood.universe, [0, 0, 0.5])
    spill_likelihood['likely'] = mf.trimf(spill_likelihood.universe, [0, 0.5, 1])
    spill_likelihood['very likely'] = mf.trimf(spill_likelihood.universe, [0.5, 0.9, 1])

    rule1 = ctrl.Rule(area['small'] & confidence['low'], spill_likelihood['unlikely'])
    rule2 = ctrl.Rule(area['small'] & confidence['medium'], spill_likelihood['unlikely'])
    rule3 = ctrl.Rule(area['small'] & confidence['high'], spill_likelihood['likely'])
    rule4 = ctrl.Rule(area['medium'] & confidence['low'], spill_likelihood['unlikely'])
    rule5 = ctrl.Rule(area['medium'] & confidence['medium'], spill_likelihood['likely'])
    rule6 = ctrl.Rule(area['medium'] & confidence['high'], spill_likelihood['very likely'])
    rule7 = ctrl.Rule(area['large'] & confidence['low'], spill_likelihood['likely'])
    rule8 = ctrl.Rule(area['large'] & confidence['medium'], spill_likelihood['very likely'])
    rule9 = ctrl.Rule(area['large'] & confidence['high'], spill_likelihood['very likely'])

    spill_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    spill = ctrl.ControlSystemSimulation(spill_ctrl)

    return spill

def load_model(model_path):
    model = detection.maskrcnn_resnet50_fpn(weights=None, num_classes=2)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([T.ToTensor()])
    image = transform(image).unsqueeze(0)
    return image

def run_inference(model, image):
    with torch.no_grad():
        predictions = model(image)
    return predictions

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

def calculate_fuzzy_likelihood(area, confidence, fuzzy_ctrl):
    area = min(max(area, 0), 10000)
    confidence = min(max(confidence, 0), 1)

    fuzzy_ctrl.input['area'] = area
    fuzzy_ctrl.input['confidence'] = confidence
    fuzzy_ctrl.compute()

    return fuzzy_ctrl.output['spill_likelihood']

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
