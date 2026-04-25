import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.efficientnet import DeepfakeEfficientNet
from utils.dataset import DeepfakeDataset, get_train_transforms, get_val_transforms
from tqdm import tqdm
import os

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    return running_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / len(loader), correct / total

def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    epochs = 10
    lr = 1e-4
    
    # Model
    model = DeepfakeEfficientNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Data
    train_dataset = DeepfakeDataset('data/train', transform=get_train_transforms())
    val_dataset = DeepfakeDataset('data/val', transform=get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Starting training on {device}...")
    
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved Best Model!")

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('data/train/real', exist_ok=True)
    os.makedirs('data/train/fake', exist_ok=True)
    os.makedirs('data/val/real', exist_ok=True)
    os.makedirs('data/val/fake', exist_ok=True)
    # main() # Uncomment to run
