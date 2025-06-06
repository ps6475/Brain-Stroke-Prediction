import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, AdamW
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Dataset Preparation
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root="C:\\Users\\HP\\Downloads\\daa data set\\DAA- IMAGE DATASET\\Brain_Stroke_CT-SCAN_image\\Train", transform=transform)
val_dataset = datasets.ImageFolder(root="C:\\Users\\HP\\Downloads\\daa data set\\DAA- IMAGE DATASET\\Brain_Stroke_CT-SCAN_image\\Validation", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print("Classes:", train_dataset.classes)

# -------------------------------
# Load Pre-trained ViT
# -------------------------------
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(train_dataset.classes)
)
model.to(device)

# -------------------------------
# Optimizer & Loss
# -------------------------------
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# -------------------------------
# Training Loop
# -------------------------------
EPOCHS = 5

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images).logits
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")

    # ---------------------------
    # Validation & Metrics Calculation
    # ---------------------------
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy()[:, 1])  # Assuming binary classification

            correct += (preds == labels).sum().item()
            total += labels.size(0)
  
    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    auc_roc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape == (2, 2) else 0  # For binary classification

    print(f"Validation Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}, Recall (Sensitivity): {recall:.4f}, F1 Score: {f1:.4f}")
    print(f"Specificity: {specificity:.4f}, AUC-ROC: {auc_roc:.4f}")

# -------------------------------
# Save the Model
# -------------------------------
torch.save(model.state_dict(), "brain_stroke_vit.pth")
print("Model saved as brain_stroke_vit.pth")
