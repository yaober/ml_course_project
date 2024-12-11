# %%
# Import necessary libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import pydicom
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} device")

# %%
def save_metrics_to_csv(metrics, file_name='training_metrics_more_epochs.csv'):
    df = pd.DataFrame([metrics])
    if os.path.exists(file_name):
        df.to_csv(file_name, mode='a', header=False, index=False)
    else:
        df.to_csv(file_name, index=False)

# %%
def preprocess(path):
    df = pd.read_csv(path)
    df[['ImageID', 'Subtype']] = df['ID'].str.rsplit('_', n=1, expand=True)
    df = df.groupby(['ImageID', 'Subtype'])['Label'].max().unstack(fill_value=0).reset_index()
    for col in df.columns[1:]:
        df[col] = df[col].astype('float32')
    return df

#Define a DicomDataset class
class DicomDataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform

    def load_dcm(self, img_path):
        try:
            dcm = pydicom.read_file(img_path)
            img = dcm.pixel_array.astype(np.float32)
            img_min, img_max = np.min(img), np.max(img)
            if img_max - img_min != 0:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img, dtype=np.float32)
            return np.expand_dims(img, axis=0)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
            return None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f'{img_id}.dcm')
        img = self.load_dcm(img_path)
        if img is None:
            raise ValueError(f"Corrupted DICOM file: {img_id}")
        if self.transform:
            img = self.transform(torch.tensor(img))
        label = torch.tensor(self.df.iloc[idx, 1:].values.astype('float32'))
        return img, label

# %%
# Custom Weighted Logarithmic Loss By Kaggle 
class WeightedLogLoss(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.epsilon = 1e-15
        self.weights = torch.tensor(weights, dtype=torch.float32).to(device) if weights else None

    def forward(self, preds, targets):
        preds = torch.clamp(preds, min=self.epsilon, max=1 - self.epsilon)
        log_loss = - (targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
        if self.weights is not None:
            log_loss *= self.weights
        return log_loss.mean()

# %%
model = models.resnet34(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, 6)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs for training.")
    model = nn.DataParallel(model)
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# %%
# Evaluation Function adn Logging
def evaluate_model(model, data_loader, criterion, epoch, model_save_path='best_model.pth'):
    model.eval()
    y_true, y_pred = [], []
    val_loss = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = torch.sigmoid(model(inputs))
            val_loss += criterion(outputs, labels).item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    accuracy = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    sensitivity = recall_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
    specificity = precision_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')

    print(f"\nValidation Loss: {val_loss / len(data_loader):.4f}")
    print(f"Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, ROC AUC: {roc_auc:.4f}")

    metrics = {
        'epoch': epoch,
        'val_loss': val_loss / len(data_loader),
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'roc_auc': roc_auc,
    }
    save_metrics_to_csv(metrics)

    # Save Model
    if epoch == 1 or metrics['roc_auc'] > evaluate_model.best_roc_auc:
        print(f"Saving the best model (ROC AUC: {metrics['roc_auc']:.4f})...")
        torch.save(model.state_dict(), model_save_path)
        evaluate_model.best_roc_auc = metrics['roc_auc']

    return metrics['val_loss'], metrics['accuracy'], metrics['sensitivity'], metrics['specificity'], metrics['roc_auc']

# Initialize Best ROC AUC
evaluate_model.best_roc_auc = 0.0

# %%
# Training F
def train_model(model, train_loader, val_loader, n_epochs=5, model_save_path='best_model.pth'):
    weights = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    criterion = WeightedLogLoss(weights=weights)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    scaler = GradScaler()

    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch {epoch}/{n_epochs}")
        model.train()
        train_loss = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Enable Mixed Precision__useless for saving time
            with autocast():
                outputs = torch.sigmoid(model(inputs))
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        # Metrics
        val_loss, accuracy, sensitivity, specificity, roc_auc = evaluate_model(
            model, val_loader, criterion, epoch, model_save_path
        )

        print(f"\nTrain Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, ROC AUC: {roc_auc:.4f}")

# %%
TRAIN_PATH = './stage_2_train'
CSV_PATH = './stage_2_train.csv'

df = preprocess(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = DicomDataset(TRAIN_PATH, train_df, transform=transform)
val_dataset = DicomDataset(TRAIN_PATH, val_df, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, 
    num_workers=32, pin_memory=True, prefetch_factor=4, persistent_workers=True
)

val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, 
    num_workers=32, pin_memory=True, prefetch_factor=4, persistent_workers=True
)

history = train_model(model, train_loader, val_loader, n_epochs=1)
