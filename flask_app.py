from flask import Flask, render_template, request, send_from_directory
import os
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import transforms
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import matplotlib.pyplot as plt
from matplotlib import cm
from io import BytesIO
import base64

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, progress=True)
model.fc = nn.Linear(model.fc.in_features, 14, bias=True)
model.load_state_dict(torch.load("best_transfer_learning_model.pth", weights_only=True))
model.eval()

labels = {
    0: "Atelectasis",
    1: "Cardiomegaly",
    2: "Effusion",
    3: "Infiltration",
    4: "Mass",
    5: "Nodule",
    6: "Pneumonia",
    7: "Pneumothorax",
    8: "Consolidation",
    9: "Edema",
    10: "Emphysema",
    11: "Fibrosis",
    12: "Pleural",
    13: "Hernia"
}

# Preprocessing function
def preprocess_data(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = np.float32(image) / 255.0
    image = Image.fromarray(image)
    transformed_image = preprocess(image)

    return transformed_image.unsqueeze(0)

def predict(image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted = torch.sigmoid(outputs)
        predicted = predicted.squeeze().cpu().numpy()
        
        label_keys = list(labels.keys())
        label_values = list(labels.values())
        print(predicted)
        predicted_label_keys = [label_keys[i] for i in range(len(predicted)) if predicted[i] > 0.001]
        predicted_labels = [label_values[i] for i in range(len(predicted)) if predicted[i] > 0.001]
        return predicted_labels, predicted_label_keys
    
def generate_gradcam_image(image_path):
    example_image = Image.open(image_path).convert("L")
    grad_cam_image = np.float32(example_image) / 255.0
    grad_cam_image = Image.fromarray(grad_cam_image)

    target_layers = [model.layer4[-1]]
    input_tensor = preprocess_data(grad_cam_image)

    labels = [i for i in range(0, 14)]
    targets = [ClassifierOutputTarget(label) for label in labels]

    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]

        img = example_image.resize((224, 224))
        img = np.float32(img) / 255.0
        img = np.repeat(img[..., np.newaxis], 3, axis=-1)
        visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    pil_img = Image.fromarray((visualization * 255).astype(np.uint8)).resize((1024, 1024))
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return img_str

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_and_display():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            return render_template('index.html', filename=file.filename, message="Upload successful! Please click 'Analyze' to predict.")

    return render_template('index.html', filename=None)

@app.route('/uploads/<filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze', methods=['POST'])
def analyze_image():
    filename = request.form.get('filename')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
    image = Image.open(filepath).convert("L")
    numpy_image = np.float32(image) / 255.0
    preprocessed_image = preprocess_data(numpy_image)
    
    predicted_labels, predicted_label_keys = predict(preprocessed_image)
    
    if predicted_labels == []:
        predicted_labels = ["No Finding"]

    prediction_message = f"Predicted Labels: {', '.join(predicted_labels)}"
    
    gradcam_img_str = generate_gradcam_image(filepath)
    gradcam_filename = 'gradcam_' + filename
    gradcam_filepath = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
    
    with open(gradcam_filepath, 'wb') as f:
        f.write(base64.b64decode(gradcam_img_str))
    
    return render_template('index.html', filename=filename, gradcam_filename=gradcam_filename, message=prediction_message)

if __name__ == '__main__':
    app.run(debug=True)
