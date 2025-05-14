import json
import os
from timeit import default_timer as timer

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms
from dotenv import load_dotenv
import boto3

# --- Setup ---
load_dotenv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

# --- Download from S3 (CPU models only) ---
def download_from_s3():
    BUCKET_NAME = 'abhiya-mlops-project'
    ARTIFACTS = [
        'checkpoints/pths/sports_cpu.pt',
        'checkpoints/pths/vegfruits_cpu.pt',
        'checkpoints/classnames/sports.json',
        'checkpoints/classnames/vegfruits.json'
    ]
    os.makedirs("checkpoints/pths", exist_ok=True)
    os.makedirs("checkpoints/classnames", exist_ok=True)
    s3 = boto3.client(
        "s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name="ap-south-1"
    )
    for artifact in ARTIFACTS:
        if not os.path.exists(artifact):
            print(f"Downloading {artifact}")
            s3.download_file(BUCKET_NAME, artifact, artifact)

# --- Image Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load models ---
def load_model(name):
    path = f"checkpoints/pths/{name}_cpu.pt"
    model = torch.jit.load(path)
    model.to(device)
    model.eval()
    return model

# --- Load class mappings ---
def load_classnames(name):
    with open(f"checkpoints/classnames/{name}.json") as f:
        mapping = json.load(f)
    return {int(v): k for k, v in mapping.items()}  # Ensure keys are integers

# --- Predict functions ---
@torch.no_grad()
def predict(img: Image.Image, model, idx2lbl):
    try:
        start = timer()
        img = transform(img).to(device)
        logits = model(img.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        top5 = torch.topk(probs, 5)
        preds = {idx2lbl[i.item()]: round(v.item(), 4) for v, i in zip(top5.values[0], top5.indices[0])}
        return preds, round(timer() - start, 4)
    except Exception as e:
        print(f"Prediction error: {e}")
        return {"Error": 0.0}, 0.0

# --- App logic ---
def main():
    download_from_s3()

    smodel = load_model("sports")
    vfmodel = load_model("vegfruits")
    sidx2lbl = load_classnames("sports")
    vfidx2lbl = load_classnames("vegfruits")

    def sports_fn(img):
        if img is None:  # Handle the case where no image is provided
            return {"No image provided": 1.0}, 0.0
        return predict(img, smodel, sidx2lbl)
    
    def veg_fn(img):
        if img is None:  # Handle the case where no image is provided
            return {"No image provided": 1.0}, 0.0
        return predict(img, vfmodel, vfidx2lbl)

    sports_interface = gr.Interface(
        fn=sports_fn,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes=5),
            gr.Number(label="Prediction Time (s)")
        ],
        title="Sports Classifier",
        cache_examples=False
    )

    veg_interface = gr.Interface(
        fn=veg_fn,
        inputs=gr.Image(type="pil"),
        outputs=[
            gr.Label(num_top_classes=5),
            gr.Number(label="Prediction Time (s)")
        ],
        title="VegFruits Classifier",
        cache_examples=False
    )

    demo = gr.TabbedInterface(
        interface_list=[sports_interface, veg_interface],
        tab_names=["Sports", "VegFruits"]
    )

    demo.launch(share=True)

if __name__ == "__main__":
    main()