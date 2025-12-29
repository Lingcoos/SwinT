import numpy as np
import pandas as pd
import os
from pathlib import Path

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models

import timm
from timm.loss import LabelSmoothingCrossEntropy # This is better than normal nn.CrossEntropyLoss

from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import sys
from tqdm import tqdm
import time
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_classes(data_dir):
    data_dir = Path(data_dir)
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes

def get_data_loaders(data_dir, batch_size, train = False):
    data_dir = Path(data_dir)
    if train:
        #train
        transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(torch.nn.ModuleList([T.ColorJitter()]), p=0.25),
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
            T.RandomErasing(p=0.1, value='random')
        ])
        train_data = datasets.ImageFolder(data_dir / "train", transform = transform)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return train_loader, len(train_data)
    else:
        # val/test
        transform = T.Compose([ # We dont need augmentation for test transforms
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD), # imagenet means
        ])
        val_data = datasets.ImageFolder(data_dir / "valid", transform=transform)
        test_data = datasets.ImageFolder(data_dir / "test", transform=transform)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=4)
        return val_loader, test_loader, len(val_data), len(test_data)


HUB_URL = "SharanSMenon/swin-transformer-hub:main"
MODEL_NAME = "swin_tiny_patch4_window7_224"


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print("-"*10)
        
        for phase in ['train', 'val']: # We do training and validation phase per epoch
            if phase == 'train':
                model.train() # model to training mode
            else:
                model.eval() # model to evaluate
            
            running_loss = 0.0
            running_corrects = 0.0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'): # no autograd makes validation go faster
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # used for accuracy
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'train':
                scheduler.step() # step at end of epoch
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc =  running_corrects.double() / dataset_sizes[phase]
            
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) # keep the best validation accuracy model
        print()
    time_elapsed = time.time() - since # slight error
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print("Best Val Acc: {:.4f}".format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model


def plot_confusion_matrix(cm, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', fontsize=6)
    plt.setp(ax.get_yticklabels(), fontsize=6)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def evaluate_split(model, loader, split_name, criterion, class_names):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    top3_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"{split_name} inference"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)

            preds = outputs.argmax(dim=1)
            running_corrects += torch.sum(preds == labels).item()

            k = min(3, outputs.size(1))
            topk = torch.topk(outputs, k=k, dim=1).indices
            top3_corrects += topk.eq(labels.view(-1, 1)).any(dim=1).float().sum().item()

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    split_size = len(loader.dataset)
    avg_loss = running_loss / split_size
    top1_acc = running_corrects / split_size
    top3_acc = top3_corrects / split_size

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    metrics_dir = Path("output")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    cm_path = metrics_dir / f"{split_name}_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, f"{split_name.title()} Confusion Matrix", cm_path)

    per_class_support = cm.sum(axis=1)
    per_class_acc = np.divide(
        cm.diagonal(),
        per_class_support,
        out=np.zeros_like(per_class_support, dtype=float),
        where=per_class_support != 0
    )
    per_class_df = pd.DataFrame({
        "class": class_names,
        "support": per_class_support,
        "per_class_accuracy": per_class_acc
    })
    perf_path = metrics_dir / f"{split_name}_per_class_metrics.csv"
    per_class_df.sort_values("support", ascending=False).to_csv(perf_path, index=False)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    report_path = metrics_dir / f"{split_name}_classification_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"Saved confusion matrix to {cm_path} and per-class metrics to {perf_path}.")

    return {
        "loss": avg_loss,
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "confusion_matrix": cm,
        "report": report,
    }


def build_model(num_classes: int):
    model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    n_inputs = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model.to(device)


def main():
    dataset_path = Path("./datasets/Butterfly-100")

    train_loader, train_len = get_data_loaders(dataset_path, 128, train=True)
    val_loader, test_loader, valid_len, test_len = get_data_loaders(dataset_path, 32, train=False)

    classes = get_classes(dataset_path / "train")
    print(classes, len(classes))

    dataloaders = {"train": train_loader, "val": val_loader}
    dataset_sizes = {"train": train_len, "val": valid_len}

    print(len(train_loader), len(val_loader), len(test_loader))
    print(train_len, valid_len, test_len)

    model = build_model(len(classes))
    print(model.head)

    criterion = LabelSmoothingCrossEntropy().to(device)
    optimizer = optim.AdamW(model.head.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)

    model_ft = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        dataloaders,
        dataset_sizes,
        num_epochs=10,
    )

    val_metrics = evaluate_split(model_ft, val_loader, "validation", criterion, classes)
    test_metrics = evaluate_split(model_ft, test_loader, "test", criterion, classes)

    print(
        "Validation -> Loss: {:.4f}, Top-1 Acc: {:.4f}, Top-3 Acc: {:.4f}".format(
            val_metrics["loss"], val_metrics["top1_acc"], val_metrics["top3_acc"]
        )
    )
    print(
        "Test -> Loss: {:.4f}, Top-1 Acc: {:.4f}, Top-3 Acc: {:.4f}".format(
            test_metrics["loss"], test_metrics["top1_acc"], test_metrics["top3_acc"]
        )
    )

    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "swin_butterfly_model.pth"
    torch.save(model_ft.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
