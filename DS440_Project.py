# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.models as models
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score, average_precision_score
from medmnist import ChestMNIST
from PIL import Image
import cv2

# %%
def load_data(data_split, image_nxn_size, n_observations):
    data = ChestMNIST(split=data_split, download=True, size=image_nxn_size)

    if n_observations > 0:
        images = data.imgs[0:n_observations]
        labels = data.labels[0:n_observations]
    else:      
        images = data.imgs
        labels = data.labels

    del data

    return images, labels

# %%
train_images, train_labels = load_data(data_split="train", image_nxn_size=28, n_observations=0)
validation_images, validation_labels = load_data(data_split="val", image_nxn_size=28, n_observations=0)
test_images, test_labels = load_data(data_split="test", image_nxn_size=28, n_observations=0)

# %%
raw_image = train_images[0]
pixels = raw_image.flatten()

plt.hist(pixels, bins=50, alpha=0.7, color='blue', edgecolor='black', label=f'Pixel Mean: {np.mean(pixels):.4f} | Std Dev: {np.std(pixels):.4f}')
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.show()


# %%
# def preprocess_data(image_set):
#     preprocess = transforms.Compose([
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#     ])

#     transformed_images = []

#     for image in tqdm.tqdm(image_set):
#         image = np.float32(image) / 255.0
#         image = Image.fromarray(image)
#         transformed_images.append(preprocess(image))

#     return torch.stack(transformed_images)

# %%
# x_train_tensor = preprocess_data(train_images)
# x_validation_tensor = preprocess_data(validation_images)
# x_test_tensor = preprocess_data(test_images)

# %%
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224), interpolation=Image.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def preprocess_images(images):
    transformed_images = []
    for img in images:
        img = Image.fromarray(img.astype(np.uint8))
        transformed_img = data_transform(img)
        transformed_images.append(transformed_img)

    return torch.stack(transformed_images)

preprocessed_images = preprocess_images(train_images)


# %%
raw_image = preprocessed_images[0]
pixels = raw_image.flatten()

mean_pixel = torch.mean(pixels).item()
std_pixel = torch.std(pixels).item()

plt.hist(pixels.numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black', label=f'Pixel Mean: {mean_pixel:.4f} | Std Dev: {std_pixel:.4f}')
plt.legend(loc='upper center')
plt.title('Distribution of Pixel Intensities in the Image')
plt.xlabel('Pixel Intensity')
plt.ylabel('Number of Pixels')
plt.show()


# %%
x_train_tensor = preprocess_images(train_images)
x_validation_tensor = preprocess_images(validation_images)
x_test_tensor = preprocess_images(test_images)

# %%
y_train_tensor = torch.tensor(train_labels)
y_validation_tensor = torch.tensor(validation_labels)
y_test_tensor = torch.tensor(test_labels)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(x_validation_tensor, y_validation_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=128, shuffle=False)  
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)  

# %%
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT, progress=True)

if True:
    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer3.parameters():
        param.requires_grad = True

    for param in model.layer4.parameters():
        param.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, train_labels.shape[1], bias=True)

# %%
num_epochs = 20
optimizer = optim.Adam(model.parameters(), lr = 0.1)
scheduler = StepLR(optimizer, step_size = 5, gamma = 0.5)

#class_counts = torch.sum(y_train_tensor, dim=0).float()
#pos_weight = 1.0 / class_counts
criterion = nn.BCEWithLogitsLoss()


# %%
best_model_path = "best_transfer_learning_model.pth"
best_loss = np.inf
best_score = 0
best_epoch = 0
sigmoid_threshold = 0.5

for epoch in range(num_epochs):
    model.train()
    
    for inputs, targets in tqdm.tqdm(train_loader, desc="Training: "):
        optimizer.zero_grad()
        outputs = model(inputs)
        targets = targets.float()

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    all_targets = []
    all_predictions = []
    validation_loss = 0.0
    validation_accuracy = 0.0
    
    with torch.no_grad():
        for validation_inputs, validation_targets in validation_loader:
            validation_outputs = model(validation_inputs)
            validation_targets = validation_targets.float()
            validation_loss += criterion(validation_outputs, validation_targets)

            probabilities = torch.sigmoid(validation_outputs)
            predictions = (probabilities > sigmoid_threshold).float()

            all_targets.extend(validation_targets)
            all_predictions.extend(probabilities)

    validation_score = roc_auc_score(all_targets, all_predictions, average="macro")
    validation_loss /= len(validation_loader.dataset)

    if validation_loss < best_loss:
        best_loss = validation_loss
    
    if validation_score > best_score:
        best_score = validation_score
        torch.save(model.state_dict(), best_model_path)

    print(f"Epoch: {epoch + 1}, Validation Loss: {validation_loss}, ROC AUC: {validation_score}")
    scheduler.step()

# %%
model.load_state_dict(torch.load("best_transfer_learning_model.pth", weights_only=True))
model.eval()

all_targets = []
all_predictions = []
test_loss = 0.0

with torch.no_grad():
    for test_inputs, test_targets in tqdm.tqdm(test_loader, desc="Testing:"):
        test_outputs = model(test_inputs)
        test_targets = test_targets.float()
        test_loss += criterion(test_outputs, test_targets)

        probabilities = torch.sigmoid(test_outputs)
        predictions = (probabilities > sigmoid_threshold).float()

        all_targets.extend(test_targets)
        all_predictions.extend(predictions)

print(all_predictions)
test_loss /= len(test_loader.dataset)
test_hamming_loss = hamming_loss(all_targets, all_predictions)
test_accuracy = accuracy_score(all_targets, all_predictions)
test_precision = precision_score(all_targets, all_predictions, average="macro")
test_recall = recall_score(all_targets, all_predictions, average="macro")
test_f1_score = f1_score(all_targets, all_predictions, average="macro")
test_roc_auc = roc_auc_score(all_targets, all_predictions, average="macro")

print(f"Test Loss: {test_loss}")
print(f"Hamming Loss: {test_hamming_loss}")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1_score}")
print(f"ROC AUC: {test_roc_auc}")


# %%
model.load_state_dict(torch.load("best_transfer_learning_model.pth", weights_only=True))
model.eval()

target_layers = [model.layer4[-1]]

image = Image.open("./Example Images/Infiltration.PNG").convert("RGB")
input_tensor = data_transform(image).unsqueeze(0)

targets = [ClassifierOutputTarget(i) for i in range(0, 15)]

with GradCAM(model=model, target_layers=target_layers) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    rgb_img = np.array(image) / 255.0
    rgb_img = cv2.resize(rgb_img, (224, 224))

    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    plt.imshow(visualization)
    plt.axis('off')
    plt.show()

# %%
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

target_layers = [model.layer4[-1]]

image = Image.open("./Example Images/Atelectasis_Effusion_Infiltration.PNG").convert("RGB")
input_tensor = data_transform(image).unsqueeze(0)

targets = [ClassifierOutputTarget(i) for i in range(0, 14)]

fig, axes = plt.subplots(3, 5, figsize=(15, 9))
axes = axes.flatten()

for idx, target in enumerate(targets):
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=[target])
        grayscale_cam = grayscale_cam[0, :]

        rgb_img = np.array(image) / 255.0
        rgb_img = cv2.resize(rgb_img, (224, 224))

        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        ax = axes[idx]
        ax.imshow(visualization)
        ax.axis('off')
        ax.set_title(f"{labels[idx]}")

for j in range(len(targets), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


