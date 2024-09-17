import torch
import torchvision.models.detection as detection
import torchvision.transforms as T
import cv2

mp = 'model/SAR_model.pth'

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