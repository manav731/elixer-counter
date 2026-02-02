import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import numpy as np

MODEL_PATH = r"C:\ElixerTracker\models\card_classifier.pth"
IMG_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
CLASS_NAMES = checkpoint["classes"]
NUM_CLASSES = len(CLASS_NAMES)

# Build model
model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)

for p in model.parameters():
    p.requires_grad = False

model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

# Preprocessing (same as validation)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_card(bgr_img):
    """
    bgr_img: OpenCV image (numpy array, BGR)
    returns: (card_name, confidence)
    """
    rgb = bgr_img[:, :, ::-1]  # BGR -> RGB
    pil_img = Image.fromarray(rgb)

    x = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)

    return CLASS_NAMES[idx.item()], conf.item()
