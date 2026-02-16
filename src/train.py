import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from src.utils import plot_confusion_matrix
import os

CLASSES = ['clean', 'muddy', 'polluted']

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(42)
if device == "cuda":
    torch.cuda.manual_seed_all(42)

def train_model(model, train_loader, test_loader, epochs=5, lr=1e-3, save_path="../artifacts"):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model_wts = None
    best_val_acc = 0.0
    history = {
        'train_acc' : [],
        'train_loss': [],
        'val_acc': [],
        'val_loss': []
    }
    
    for epoch in range(1, epochs+1):
        # training
        model.train()
        running_loss = correct = total = 0

        for imgs, labels in tqdm(train_loader, desc=f'Epoch {epoch} [Train]'):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        history['train_acc'].append(train_acc)
        history['train_loss'].append(train_loss)

        # validation
        model.eval()
        val_loss = correct = total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f'Epoch {epoch} [Val]'):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(test_loader)
        val_acc = 100 * correct / total
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)
        
        print(f'Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.2f}% | Val Loss {val_loss:.4f}, Val Acc {val_acc:.2f}%')
        if(val_acc > best_val_acc):
            best_val_acc = val_acc
            best_model_wts = model.state_dict().copy()
            os.makedirs(save_path, exist_ok=True)
            model_path = os.path.join(save_path, "best_model.pth")
            torch.save(best_model_wts, model_path)


    return model, history

def evaluate_model(model, test_loader):
    model = model.to(device)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    print(classification_report(all_labels, all_preds, target_names=CLASSES))

    # Plot Confusion Matrix
    plot_confusion_matrix(all_labels, all_preds)
