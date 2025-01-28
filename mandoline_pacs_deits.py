import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from timm import create_model
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from PIL import Image

# Dataset Directory
PACS_DIR = ".data/pacs_data/pacs_data"

# Custom PACS Dataset with Slices
class SlicedPACS(torch.utils.data.Dataset):
    def __init__(self, dataset, slice_fn):
        self.dataset = dataset
        self.slice_fn = slice_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        img = transforms.ToTensor()(img)
        slice_label = self.slice_fn(label)
        return img, label, slice_label  # Include slice label

# Slice Function
def construct_slice_representation(labels):
    """
    Constructs slice representations for vehicles vs animals and small vs large objects.
    """
    # Example slice logic for PACS:
    # Vehicle vs Animal
    vehicle_classes = [0, 1, 8, 9]  # Placeholder, adjust based on PACS classes
    even_odd_slice = np.isin(labels, vehicle_classes).astype(int)  # 1 for vehicles, 0 for animals

    # Small vs Large Object (example logic, replace with real PACS rules)
    small_classes = [0, 2, 4]
    small_large_slice = np.isin(labels, small_classes).astype(int)  # 1 for small objects, 0 for large

    return np.stack([even_odd_slice, small_large_slice], axis=1)

# Load PACS Dataset
def load_pacs_data(domain, batch_size, slice_fn=None, corruption_fn=None):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset = torchvision.datasets.ImageFolder(os.path.join(PACS_DIR, domain), transform=transform)

    if slice_fn:
        dataset = SlicedPACS(dataset, slice_fn)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize DeiT-S Model
def initialize_model(num_classes=7):
    # Create a DeiT-S model with timm
    model = create_model("deit_small_patch16_224", pretrained=True)
    model.head = nn.Linear(model.head.in_features, num_classes)  # Adjust output layer for 7 classes
    return model

# Train Model
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluate Model
def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

# Compute Empirical Matrix
def get_correct(model, dataloader, device):
    correct_predictions = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct = (preds == labels).long()
            correct_predictions.append(correct.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    return np.concatenate(correct_predictions), np.concatenate(all_labels)

# Estimate Performance
def estimate_performance(D_src, D_tgt, empirical_mat_list_src):
    n_slices = D_src.shape[1]
    performance_estimates = []
    for slice_idx in range(n_slices):
        src_slice = D_src[:, slice_idx]
        slice_accuracy = np.mean(empirical_mat_list_src[0][src_slice == 1])
        performance_estimates.append(slice_accuracy)
    return np.mean(performance_estimates)

# Corruption Function: Gaussian Noise
def gaussian_noise(image, severity=3):
    noise = np.random.normal(0, severity, image.shape).astype(np.float32)
    noisy_image = np.clip(image + noise, 0, 1).astype(np.float32)
    return noisy_image

# Evaluate Mean Corruption Error
def evaluate_mce(model, dataloader, corruption_fn, device, severity=3):
    all_accs = []
    for images, labels in dataloader:
        images = images.permute(0, 2, 3, 1).numpy()  # Convert to HWC
        corrupted_images = np.stack([corruption_fn(img, severity) for img in images])
        corrupted_images = torch.tensor(corrupted_images).permute(0, 3, 1, 2).to(device)  # Convert back to CHW
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(corrupted_images)
            preds = outputs.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            all_accs.append(acc)
    return 1 - np.mean(all_accs)

# Main Execution
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0001

    # Load PACS photo domain
    photo_loader = load_pacs_data("photo", batch_size)

    # Target domain loaders
    target_domains = ["art_painting", "cartoon", "sketch"]
    target_loaders = {domain: load_pacs_data(domain, batch_size) for domain in target_domains}

    # Initialize and train model
    model = initialize_model()
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        train_loss = train_model(model, photo_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")

    all_target_labels = []
    for domain, loader in target_loaders.items():  # Iterate through target loaders
        for _, labels in loader:  # Iterate through batches in each loader
            all_target_labels.extend(labels.cpu().numpy())  # Accumulate labels

    if len(all_target_labels) == 0:
        raise ValueError("No labels found in target loaders. Check the target dataset or DataLoader.")

    standard_accs = {}
    for domain, loader in target_loaders.items():
        acc = evaluate_model(model, loader, device)
        standard_accs[domain] = acc
        print(f"Domain: {domain}, Accuracy: {acc:.4f}")

    # Prepare slice representation
    photo_correct, photo_labels = get_correct(model, photo_loader, device)
    empirical_mat_list_src = [photo_correct[:, None]]
    D_src = construct_slice_representation(photo_labels)

    D_tgt = construct_slice_representation(np.array(all_target_labels))

    performance_estimate = estimate_performance(D_src, D_tgt, empirical_mat_list_src)
    print(f"Estimated Performance: {performance_estimate:.4f}")

    # Evaluate mCE
    mce = evaluate_mce(model, target_loaders["cartoon"], gaussian_noise, device)
    print(f"Mean Corruption Error (mCE): {mce:.4f}")
