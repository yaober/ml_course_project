# Import Libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import pydicom
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score

import os
import pydicom
from PIL import Image
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Convert DICOM to PNG
def convert_dicom_to_png_worker(args):
    dicom_path, output_dir = args
    try:
        dcm = pydicom.read_file(dicom_path)
        img = dcm.pixel_array.astype(np.float32)

        # Normalize image to [0, 255]
        img_min = img.min()
        img_max = img.max()
        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        # Save as PNG
        img_id = os.path.basename(dicom_path).replace(".dcm", ".png")
        output_file = os.path.join(output_dir, img_id)
        Image.fromarray(img).save(output_file)
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")


# Batch Convert DICOM to PNG with Multi-Processing
def batch_convert_to_png_mp(dicom_dir, output_dir, num_workers=72):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dicom_files = [
        os.path.join(dicom_dir, file)
        for file in os.listdir(dicom_dir)
        if file.endswith(".dcm")
    ]

    # Use all available CPUs if not specified
    num_workers = num_workers or cpu_count()

    print(f"Starting Multi-Processing with {num_workers} workers...")
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(
            convert_dicom_to_png_worker, [(file, output_dir) for file in dicom_files]
        ), total=len(dicom_files), desc="Converting DICOM to PNG"))

# Preprocess CSV Labels
def preprocess(path):
    df = pd.read_csv(path)
    df[['ImageID', 'Subtype']] = df['ID'].str.rsplit('_', n=1, expand=True)
    df = df.groupby(['ImageID', 'Subtype'])['Label'].max().unstack(fill_value=0).reset_index()
    for col in df.columns[1:]:
        df[col] = df[col].astype('float32')
    return df

# Custom Dataset for PNG Images
class PngDataset(Dataset):
    def __init__(self, img_dir, df, transform=None):
        self.img_dir = img_dir
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f'{img_id}.png')

        # Load PNG image
        img = Image.open(img_path).convert('L')  # Grayscale

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.df.iloc[idx, 1:].values.astype('float32'))
        return img, label

# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=5):
    best_roc_auc = 0.0
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        model.train()
        train_loss = 0

        for inputs, labels in tqdm(train_loader, desc="Training"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(inputs))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_loss, accuracy, sensitivity, specificity, roc_auc = evaluate_model(model, val_loader, criterion)

        print(f"\nTrain Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, ROC AUC: {roc_auc:.4f}")

        # Save the best model
        if roc_auc > best_roc_auc:
            print(f"Saving Best Model with ROC AUC: {roc_auc:.4f}")
            torch.save(model.state_dict(), "best_model.pth")
            best_roc_auc = roc_auc

# Evaluation Function
def evaluate_model(model, data_loader, criterion):
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

    accuracy = accuracy_score(y_true.argmax(axis=1), y_pred.argmax(axis=1))
    sensitivity = recall_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
    specificity = precision_score(y_true.argmax(axis=1), y_pred.argmax(axis=1), average='weighted')
    roc_auc = roc_auc_score(y_true, y_pred, average='weighted', multi_class='ovr')

    return val_loss / len(data_loader), accuracy, sensitivity, specificity, roc_auc

# Define Paths
TRAIN_DICOM_DIR = './stage_2_train'
TRAIN_PNG_DIR = './train_png'
CSV_PATH = './stage_2_train.csv'

# Convert Images
batch_convert_to_png_mp(TRAIN_DICOM_DIR, TRAIN_PNG_DIR)

# Preprocess CSV
df = preprocess(CSV_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Define Data Loaders
transform = transforms.Compose([
    transforms.Resize((128, 128), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = PngDataset(TRAIN_PNG_DIR, train_df, transform=transform)
val_dataset = PngDataset(TRAIN_PNG_DIR, val_df, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True
)

val_loader = DataLoader(
    val_dataset, batch_size=128, shuffle=False, num_workers=16, pin_memory=True
)

# Train the Model
train_model(model, train_loader, val_loader, criterion, optimizer, n_epochs=10)
