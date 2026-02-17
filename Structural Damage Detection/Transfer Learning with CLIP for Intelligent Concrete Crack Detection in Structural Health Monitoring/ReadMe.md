
# Transfer Learning With CLIP for Intelligent Concrete Crack Detection in Structural Health Monitoring 

>Paper on [Wiley](https://onlinelibrary.wiley.com/doi/10.1155/vib/4194403) / [GitHub Repo](https://github.com/MdSiamAnsary/ClassificationTasks/blob/main/Structural%20Damage%20Detection/Transfer%20Learning%20with%20CLIP%20for%20Intelligent%20Concrete%20Crack%20Detection%20in%20Structural%20Health%20Monitoring/100475884.pdf)

>Originally collected Dataset: [Structural Defects Network (SDNET) 2018 on Kaggle](https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images)

>Code

```python
import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    matthews_corrcoef
)
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()


data_dir = "SDNET2018" 

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

dataset = datasets.ImageFolder(data_dir, transform=train_transform)

train_size = int(0.7 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    dataset, [train_size, val_size, test_size]
)

val_dataset.dataset.transform = val_transform
test_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = 2


class FeatureExtractor(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.net = timm.create_model(model_name,
                                     pretrained=True,
                                     num_classes=0)

    def forward(self, x):
        return self.net(x)


class HybridClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


model_name = 'vit_base_patch16_clip_224.openai'

extractor = FeatureExtractor(model_name).to(device)

dummy_input = torch.randn(1,3,224,224).to(device)
feature_dim = extractor(dummy_input).shape[1]

classifier = HybridClassifier(feature_dim, num_classes).to(device)

params = list(extractor.parameters()) + list(classifier.parameters())

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params, lr=1e-4)

num_epochs = 10
best_val_f1 = 0
best_model_wts = None


for epoch in range(num_epochs):
    extractor.train()
    classifier.train()

    running_loss = 0

    for imgs, labels in tqdm(train_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        features = extractor(imgs)
        outputs  = classifier(features)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

    
    extractor.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            features = extractor(imgs)
            outputs = classifier(features)

            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    val_f1 = f1_score(all_labels, all_preds)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_wts = copy.deepcopy({
            "extractor": extractor.state_dict(),
            "classifier": classifier.state_dict()
        })

print("Training complete.")


torch.save(best_model_wts, "best_model.pth")


def evaluate(loader):
    extractor.eval()
    classifier.eval()

    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            features = extractor(imgs)
            outputs = classifier(features)

            probs = torch.softmax(outputs, dim=1)[:,1]
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    mcc = matthews_corrcoef(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    print("\nTest Results:")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)
    print("ROC-AUC:", auc)
    print("MCC:", mcc)
    print("Confusion Matrix:\n", cm)

evaluate(test_loader)


from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

target_layer = extractor.net.blocks[-1].norm1

cam = GradCAM(model=extractor, target_layers=[target_layer], use_cuda=True)

sample_img, _ = dataset[0]
input_tensor = sample_img.unsqueeze(0).to(device)

grayscale_cam = cam(input_tensor=input_tensor)[0]

img_np = sample_img.permute(1,2,0).numpy()
img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.axis('off')
plt.title("Grad-CAM Visualization")
plt.show()
```


