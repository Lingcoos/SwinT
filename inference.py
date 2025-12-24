import numpy as np
import pandas as pd
import os
from pathlib import Path

import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

import timm
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_path, num_classes):
    """load the trained model"""

    HUB_URL = "SharanSMenon/swin-transformer-hub:main"
    MODEL_NAME = "swin_tiny_patch4_window7_224"
    
    model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=False)
    

    n_inputs = model.head.in_features
    model.head = nn.Sequential(
        nn.Linear(n_inputs, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    return model

def get_test_loader(data_dir, batch_size=32):
    """Create test dataset loader"""
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(timm.data.IMAGENET_DEFAULT_MEAN, timm.data.IMAGENET_DEFAULT_STD),
    ])
    
    test_data = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return test_loader, test_data.classes

def evaluate_model(model, test_loader, class_names):
    """Evaluate model performance, calculating top-1 and top-3 accuracy"""
    model.eval()
    
    total_samples = 0
    top1_correct = 0
    top3_correct = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Inferencing"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate top-1 accuracy
            _, predicted = torch.max(outputs.data, 1)
            top1_correct += (predicted == labels).sum().item()
            
            # Calculate top-3 accuracy
            _, top3_pred = outputs.topk(3, 1, True, True)
            top3_correct += torch.sum(top3_pred.eq(labels.view(-1, 1)).any(dim=1)).item()
            
            total_samples += labels.size(0)
            
            # Save predictions for analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    top1_accuracy = 100.0 * top1_correct / total_samples
    top3_accuracy = 100.0 * top3_correct / total_samples
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_predictions
    })
    
    # Convert numeric labels to class names
    results_df['true_class'] = results_df['true_label'].apply(lambda x: class_names[x])
    results_df['predicted_class'] = results_df['predicted_label'].apply(lambda x: class_names[x])
    
    # Calculate accuracy for each class
    class_stats = []
    for i, class_name in enumerate(class_names):
        class_samples = results_df[results_df['true_label'] == i]
        if len(class_samples) > 0:
            class_correct = (class_samples['true_label'] == class_samples['predicted_label']).sum()
            class_accuracy = 100.0 * class_correct / len(class_samples)
            class_stats.append({
                'class_name': class_name,
                'samples': len(class_samples),
                'correct': int(class_correct),
                'accuracy': class_accuracy
            })
    
    class_stats_df = pd.DataFrame(class_stats)
    
    return top1_accuracy, top3_accuracy, results_df, class_stats_df

def print_confusion_matrix(results_df, class_names, top_n=10):
    """Print confusion matrix (showing only top-N classes)"""
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(results_df['true_label'], results_df['predicted_label'])
    
    # Show only top-N classes with the most samples
    class_counts = results_df['true_label'].value_counts().sort_values(ascending=False)
    top_classes_idx = class_counts.head(top_n).index.tolist()
    top_class_names = [class_names[i] for i in top_classes_idx]
    
    print(f"\nConfusion Matrix (Top-{top_n} Classes):")
    print("-" * 50)
    
    # Print header
    print(f"{'True Class':<20} {'Predicted Class Distribution':<50}")
    print("-" * 50)
    
    for i, true_idx in enumerate(top_classes_idx):
        true_class = class_names[true_idx]
        row = cm[true_idx][top_classes_idx]
        
        # 创建分布字符串
        dist_str = ""
        for j, pred_idx in enumerate(top_classes_idx):
            count = cm[true_idx][pred_idx]
            if count > 0:
                dist_str += f"{class_names[pred_idx]}:{count}  "
        
        print(f"{true_class:<20} {dist_str:<50}")
    
    print("-" * 50)

def main():
    # 设置路径
    model_path = "models/swin_butterfly_model.pth"
    test_data_dir = "datasets/Butterfly-100/test"
    
    # 检查路径是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist!")
        return
    
    if not os.path.exists(test_data_dir):
        print(f"Error: Test data directory '{test_data_dir}' does not exist!")
        return
    
    print(f"Device: {device}")
    print(f"Loading model: {model_path}")
    print(f"Test data: {test_data_dir}")
    
    # Get class information (from train folder for class order, or infer from test folder)
    # Here we try to get from test folder, if fails then use numeric labels
    try:
        train_data_dir = "datasets/Butterfly-100/train"
        train_dataset = datasets.ImageFolder(train_data_dir)
        class_names = train_dataset.classes
        num_classes = len(class_names)
        print(f"Found {num_classes} classes")
    except:
        # If unable to get train classes, infer from test set
        test_dataset = datasets.ImageFolder(test_data_dir)
        class_names = test_dataset.classes
        num_classes = len(class_names)
        print(f"Inferred {num_classes} classes from test set")
    
    # Load model
    model = load_model(model_path, num_classes)
    print("Model loaded successfully!")
    
    # Create test data loader
    test_loader, test_class_names = get_test_loader(test_data_dir)
    
    # Ensure test class order matches train class order
    if class_names != test_class_names:
        print("Warning: Test class order does not match train class order, using test class order")
        class_names = test_class_names
    
    print(f"Number of test samples: {len(test_loader.dataset)}")
    
    # Evaluate model
    top1_acc, top3_acc, results_df, class_stats_df = evaluate_model(model, test_loader, class_names)
    
    # Print results
    print("\n" + "="*50)
    print("Model Performance Evaluation Results")
    print("="*50)
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    print(f"Total Samples: {len(results_df)}")
    print(f"Correct Predictions (Top-1): {int(len(results_df) * top1_acc / 100)}")
    print("="*50)
    
    # Print accuracy for each class
    print("\nAccuracy for Each Class:")
    print("-"*60)
    print(f"{'Class Name':<25} {'Samples':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-"*60)
    
    for _, row in class_stats_df.iterrows():
        print(f"{row['class_name']:<25} {row['samples']:<10} {row['correct']:<10} {row['accuracy']:.2f}%")
    
    print("-"*60)
    print(f"{'Total/Average':<25} {class_stats_df['samples'].sum():<10} {class_stats_df['correct'].sum():<10} {class_stats_df['accuracy'].mean():.2f}%")
    
    # Save detailed results to CSV files
    output_dir = "output/inference"
    os.makedirs(output_dir, exist_ok=True)
    
    results_df.to_csv(f"{output_dir}/detailed_predictions.csv", index=False)
    class_stats_df.to_csv(f"{output_dir}/class_accuracy_stats.csv", index=False)
    
    # Save summary results
    summary = {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'total_samples': len(results_df),
        'correct_predictions': int(len(results_df) * top1_acc / 100),
        'model_path': model_path,
        'test_data_path': test_data_dir
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{output_dir}/summary.csv", index=False)
    
    print(f"\nDetailed results have been saved to the '{output_dir}' directory")
    
    # Print confusion matrix (top 10 classes)
    print_confusion_matrix(results_df, class_names, top_n=10)
    
    # Analyze most common errors
    print("\nMost Common Misclassifications:")
    print("-"*60)
    
    # Find samples with incorrect predictions
    wrong_predictions = results_df[results_df['true_label'] != results_df['predicted_label']]
    
    if len(wrong_predictions) > 0:
        # Count the most common error pairs
        error_counts = wrong_predictions.groupby(['true_class', 'predicted_class']).size().reset_index(name='count')
        error_counts = error_counts.sort_values('count', ascending=False).head(10)
        
        for _, row in error_counts.iterrows():
            print(f"{row['true_class']} -> {row['predicted_class']}: {row['count']} times")
    else:
        print("No misclassified samples!")

if __name__ == "__main__":
    main()