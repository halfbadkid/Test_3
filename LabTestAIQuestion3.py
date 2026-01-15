import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import requests
import pandas as pd

st.set_page_config(page_title="Webcam Image Classification", layout="centered")
st.title("📷 Real-Time Webcam Image Classification (ResNet-18)")

# Step 1 & 2: Load ImageNet labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
labels = requests.get(LABELS_URL).text.splitlines()

# Step 3: Load pretrained ResNet-18
model = models.resnet18(pretrained=True)
model.eval()

# Step 4: Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Step 5: Capture image from webcam
camera_image = st.camera_input("Capture an image using your webcam")

if camera_image is not None:
    image = Image.open(camera_image).convert("RGB")
    st.image(image, caption="Captured Image", use_container_width=True)
    input_tensor = preprocess(image).unsqueeze(0)

    # Step 6: Model inference and prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    # Top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        results.append({
            "Rank": i + 1,
            "Label": labels[top5_catid[i]],
            "Probability": round(top5_prob[i].item(), 4)
        })

    df = pd.DataFrame(results)

    st.subheader("🔍 Top 5 Prediction Results")
    st.table(df)
