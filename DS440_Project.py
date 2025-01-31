import numpy as np
import torchvision.models as models
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, roc_auc_score
from medmnist import ChestMNIST
from PIL import Image

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

def preprocess_data(image_set):
    preprocess = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformed_images = []

    for image in tqdm.tqdm(image_set):
        image = np.float32(image) / 255.0
        image = Image.fromarray(image)
        transformed_images.append(preprocess(image))

    return torch.stack(transformed_images)

train_images, train_labels = load_data(data_split="train", image_nxn_size=224, n_observations=1000)
validation_images, validation_labels = load_data(data_split="val", image_nxn_size=224, n_observations=1000)
test_images, test_labels = load_data(data_split="test", image_nxn_size=224, n_observations=1000)

x_train_tensor = preprocess_data(train_images)
x_validation_tensor = preprocess_data(validation_images)
x_test_tensor = preprocess_data(test_images)

y_train_tensor = torch.tensor(train_labels)
y_validation_tensor = torch.tensor(validation_labels)
y_test_tensor = torch.tensor(test_labels)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(x_validation_tensor, y_validation_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

# Create DataLoaders for efficient training and testing data handling
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False)  
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)  

model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT, progress=True)
n_features = 9

for param in model.features[:n_features].parameters():
    param.requires_grad = False

for param in model.features[n_features:].parameters():
    param.requires_grad = True

model.classifier[1] = nn.Conv2d(512, train_labels.shape[1], kernel_size=(1, 1), stride=(1, 1))
model.classifier[2] = nn.Identity()

optimizer = optim.Adam(model.parameters(), lr = 0.1)
#scheduler = StepLR(optimizer, step_size = 2, gamma = 0.5)
criterion = nn.BCEWithLogitsLoss()

best_model_path = "best_transfer_learning_model.pth"
best_loss = np.inf
best_accuracy = 0
best_epoch = 0
sigmoid_threshold = 0.25

for epoch in range(5):
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
            all_predictions.extend(predictions)

    validation_accuracy = accuracy_score(all_targets, all_predictions)
    validation_loss /= len(validation_loader.dataset)

    if validation_loss < best_loss:
        best_loss = validation_loss
    
    # Save the model with the best accuracy
    if validation_accuracy > best_accuracy:
        best_accuracy = validation_accuracy
        torch.save(model.state_dict(), best_model_path)

    print(f"Epoch: {epoch + 1}, Validation Loss: {validation_loss}, Accuracy: {validation_accuracy}")

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
test_loss /= len(validation_loader.dataset)
test_hamming_loss = hamming_loss(all_targets, all_predictions)
test_accuracy = accuracy_score(all_targets, all_predictions)
test_precision = precision_score(all_targets, all_predictions, average="micro")
test_recall = recall_score(all_targets, all_predictions, average="micro")
test_f1_score = f1_score(all_targets, all_predictions, average="micro")
test_roc_auc = roc_auc_score(all_targets, all_predictions)

print(f"Test Loss: {test_loss}")
print(f"Hamming Loss: {test_hamming_loss}")
print(f"Accuracy: {test_accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1 Score: {test_f1_score}")
print(f"ROC AUC: {test_roc_auc}")
