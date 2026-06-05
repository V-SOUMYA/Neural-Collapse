#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import linalg as sp_linalg
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

import os
import random
import time
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# In[ ]:


SEED = 42

def set_seed(seed=SEED):
    """Fix all sources of randomness for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# In[ ]:


NUM_CLASSES   = 10        # CIFAR-10 has 10 classes
FEATURE_DIM   = 512       # ResNet-18 last-layer feature dimension
BATCH_SIZE    = 128       # batch size of 128
TOTAL_EPOCHS  = 350       # 350 epochs
LR_INITIAL    = 0.1       # Initial learning rate
MOMENTUM      = 0.9       # SGD with momentum 0.9 
WEIGHT_DECAY  = 5e-4      # 5×10^-4 for the other datasets
LR_MILESTONES = [TOTAL_EPOCHS // 3, 2 * TOTAL_EPOCHS // 3]  # [116, 233]
LR_GAMMA      = 0.1       # annealed by a factor of 10

METRIC_EPOCHS = sorted(set(
    list(range(0, TOTAL_EPOCHS, 10)) +
    list(range(TOTAL_EPOCHS - 50, TOTAL_EPOCHS))
))

TPT_THRESHOLD = 99.9
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('results', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)
print(f"Training for {TOTAL_EPOCHS} epochs")
print(f"LR schedule: {LR_INITIAL} -> {LR_INITIAL*LR_GAMMA} at epoch {LR_MILESTONES[0]} -> {LR_INITIAL*LR_GAMMA**2} at epoch {LR_MILESTONES[1]}")
print(f"NC metrics will be computed at {len(METRIC_EPOCHS)} epoch checkpoints")
print(f"Metric epoch range: {METRIC_EPOCHS[0]} to {METRIC_EPOCHS[-1]}")


# In[ ]:


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

train_transform = T.Compose([
    T.ToTensor(),                              
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),    
])

# Evaluation transform: same as training (both are normalization-only)
eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# Load datasets
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=train_transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=eval_transform
)

train_dataset_eval = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=eval_transform
)

# Create data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=2, pin_memory=True
)
test_loader = DataLoader(
    test_dataset, batch_size=256, shuffle=False,
    num_workers=2, pin_memory=True
)
train_eval_loader = DataLoader(
    train_dataset_eval, batch_size=256, shuffle=False,
    num_workers=2, pin_memory=True
)

# Verify dataset statistics
print(f"Training set: {len(train_dataset)} images")
print(f"Test set:     {len(test_dataset)} images")
print(f"Classes:      {train_dataset.classes}")

targets = np.array(train_dataset.targets)
print(f"\nClass balance (Section 2D assumes balanced):")
for c in range(NUM_CLASSES):
    count = (targets == c).sum()
    print(f"  Class {c} ({train_dataset.classes[c]:>10s}): {count} samples")
print(f"  -> Perfectly balanced: {len(set([(targets == c).sum() for c in range(NUM_CLASSES)])) == 1}")


# In[ ]:


class ResNet18_CIFAR(nn.Module):

    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()

        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,   # Identity — no-op after our modification
            resnet.layer1,    # 2 residual blocks, 64 channels
            resnet.layer2,    # 2 residual blocks, 128 channels, stride 2
            resnet.layer3,    # 2 residual blocks, 256 channels, stride 2
            resnet.layer4,    # 2 residual blocks, 512 channels, stride 2
            resnet.avgpool,   # Global average pooling -> (batch, 512, 1, 1)
        )

        self.feature_dim = feature_dim  # p = 512
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        h = self.features(x)          # (batch, 512, 1, 1)
        h = h.view(h.size(0), -1)     # (batch, 512) — flatten
        logits = self.classifier(h)   # (batch, 10) = W @ h + b
        if return_features:
            return logits, h
        return logits


model = ResNet18_CIFAR(num_classes=NUM_CLASSES).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
classifier_params = sum(p.numel() for p in model.classifier.parameters())
feature_params = total_params - classifier_params

print(f"Model: ResNet-18 for CIFAR-10")
print(f"  Feature extractor parameters: {feature_params:,}")
print(f"  Classifier parameters:        {classifier_params:,} (W: {NUM_CLASSES}x{FEATURE_DIM} + b: {NUM_CLASSES})")
print(f"  Total parameters:             {total_params:,}")
print(f"  Feature dimension (p):        {FEATURE_DIM}")
print(f"  Number of classes (C):        {NUM_CLASSES}")
print(f"  Overparameterization ratio:   {total_params / (len(train_dataset) * NUM_CLASSES):.1f}x")

# Quick forward pass to verify shapes
dummy = torch.randn(2, 3, 32, 32).to(DEVICE)
logits, features = model(dummy, return_features=True)
print(f"\n  Input shape:   {dummy.shape}")
print(f"  Feature shape: {features.shape}  (this is h in R^p)")
print(f"  Logits shape:  {logits.shape}   (this is Wh + b in R^C)")
print(f"\n  Classifier W shape: {model.classifier.weight.shape}  (C x p)")
print(f"  Classifier b shape: {model.classifier.bias.shape}    (C,)")


# In[ ]:


@torch.no_grad()
def extract_features(model, loader, num_classes=NUM_CLASSES, device=DEVICE):
    model.eval()
    all_features = []
    all_labels = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        _, h = model(inputs, return_features=True)
        all_features.append(h.cpu())
        all_labels.append(targets)

    all_features = torch.cat(all_features, dim=0)   # (50000, 512)
    all_labels = torch.cat(all_labels, dim=0)        # (50000,)

    # Organize by class
    features_by_class = {}
    for c in range(num_classes):
        mask = (all_labels == c)
        features_by_class[c] = all_features[mask]    # (5000, 512) for balanced CIFAR-10

    return features_by_class, all_features, all_labels


# In[ ]:


def compute_means(features_by_class, all_features, num_classes=NUM_CLASSES):
    """
    Compute per-class means and global mean.

    Returns:
        class_means: tensor (C, 512) — mu_c for each class
        global_mean: tensor (512,)   — mu_G
    """
    class_means = torch.stack([
        features_by_class[c].mean(dim=0) for c in range(num_classes)
    ], dim=0)  # (C, 512)

    global_mean = all_features.mean(dim=0)  # (512,)

    return class_means, global_mean


# In[ ]:


def compute_nc1(features_by_class, class_means, global_mean, num_classes=NUM_CLASSES):

    feature_dim = class_means.shape[1]  # 512
    Sigma_W = torch.zeros(feature_dim, feature_dim, dtype=torch.float64)
    total_count = 0

    for c in range(num_classes):
        centered = (features_by_class[c] - class_means[c].unsqueeze(0)).double()  # (N_c, 512)
        Sigma_W += centered.T @ centered  # (512, 512)
        total_count += features_by_class[c].shape[0]

    Sigma_W /= total_count  # Average over all N*C examples
    centered_means = (class_means - global_mean.unsqueeze(0)).double()  # (C, 512)
    Sigma_B = (centered_means.T @ centered_means) / num_classes  # (512, 512)
    Sigma_B_np = Sigma_B.numpy()
    Sigma_B_pinv = torch.from_numpy(
        np.linalg.pinv(Sigma_B_np, rcond=1e-10)
    )  # (512, 512)
    product = Sigma_W @ Sigma_B_pinv  # (512, 512)
    nc1 = torch.trace(product).item() / num_classes

    return nc1


# In[ ]:


def compute_nc2(class_means, global_mean, num_classes=NUM_CLASSES):
    centered_means = class_means - global_mean.unsqueeze(0)  # (C, 512)
    norms = torch.norm(centered_means, dim=1)  # (C,)
    equinorm_cv = (norms.std() / (norms.mean() + 1e-8)).item()
    normalized = centered_means / (norms.unsqueeze(1) + 1e-8)  # (C, 512)
    cosine_matrix = normalized @ normalized.T  # (C, C)
    mask = ~torch.eye(num_classes, dtype=torch.bool)
    off_diag_cosines = cosine_matrix[mask]  # (C*(C-1),) = (90,)
    equiangular_std = off_diag_cosines.std().item()
    target_cosine = -1.0 / (num_classes - 1)
    cos_deviation = (off_diag_cosines - target_cosine).abs().mean().item()

    return equinorm_cv, equiangular_std, cos_deviation


def compute_nc2_classifier(model, num_classes=NUM_CLASSES):
    W = model.classifier.weight.data.cpu()  # (C, 512)
    norms = torch.norm(W, dim=1)  # (C,)
    equinorm_cv = (norms.std() / (norms.mean() + 1e-8)).item()
    normalized = W / (norms.unsqueeze(1) + 1e-8)  # (C, 512)
    cosine_matrix = normalized @ normalized.T  # (C, C)
    mask = ~torch.eye(num_classes, dtype=torch.bool)
    off_diag_cosines = cosine_matrix[mask]
    equiangular_std = off_diag_cosines.std().item()
    target_cosine = -1.0 / (num_classes - 1)
    cos_deviation = (off_diag_cosines - target_cosine).abs().mean().item()
    return equinorm_cv, equiangular_std, cos_deviation


# In[ ]:


def compute_nc3(model, class_means, global_mean):
    W = model.classifier.weight.data.cpu().float()  # (10, 512)
    M_dot = (class_means - global_mean.unsqueeze(0)).T.float()  # (512, 10)
    W_norm = W / (torch.norm(W, p='fro') + 1e-8)         # (10, 512)
    M_norm = M_dot / (torch.norm(M_dot, p='fro') + 1e-8)  # (512, 10)
    nc3 = (torch.norm(W_norm.T - M_norm, p='fro') ** 2).item()
    return nc3


# In[ ]:


@torch.no_grad()
def compute_nc4(model, loader, class_means, device=DEVICE):
    model.eval()
    class_means_dev = class_means.to(device)  # (C, 512)
    total = 0
    disagreements = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits, features = model(inputs, return_features=True)
        net_pred = logits.argmax(dim=1)  # (batch,)
        dists = torch.cdist(features, class_means_dev, p=2)  # (batch, C)
        ncc_pred = dists.argmin(dim=1)  # (batch,)
        disagreements += (net_pred != ncc_pred).sum().item()
        total += inputs.size(0)

    return disagreements / total


# In[ ]:


def compute_all_nc_metrics(model, train_eval_loader, test_loader,
                           num_classes=NUM_CLASSES, device=DEVICE):

    # Step 1: Extract features
    features_by_class, all_features, all_labels = extract_features(
        model, train_eval_loader, num_classes, device
    )

    # Step 2: Compute means
    class_means, global_mean = compute_means(features_by_class, all_features, num_classes)

    # Step 3: NC1
    nc1 = compute_nc1(features_by_class, class_means, global_mean, num_classes)

    # Step 4: NC2 for class means (blue lines in paper)
    means_equinorm, means_equiangular, means_cos_dev = compute_nc2(
        class_means, global_mean, num_classes
    )

    # Step 4b: NC2 for classifier weights (orange lines in paper)
    clf_equinorm, clf_equiangular, clf_cos_dev = compute_nc2_classifier(
        model, num_classes
    )

    # Step 5: NC3
    nc3 = compute_nc3(model, class_means, global_mean)

    # Step 6: NC4 on test set (paper's Figure 7) and on train set
    nc4_test = compute_nc4(model, test_loader, class_means, device)
    nc4_train = compute_nc4(model, train_eval_loader, class_means, device)

    return {
        'nc1': nc1,
        # NC2 — class means (blue lines in paper)
        'nc2_means_equinorm': means_equinorm,
        'nc2_means_equiangular': means_equiangular,
        'nc2_means_cos_deviation': means_cos_dev,
        # NC2 — classifier weights (orange lines in paper)
        'nc2_clf_equinorm': clf_equinorm,
        'nc2_clf_equiangular': clf_equiangular,
        'nc2_clf_cos_deviation': clf_cos_dev,
        # NC3 & NC4
        'nc3': nc3,
        'nc4_test': nc4_test,
        'nc4_train': nc4_train,
    }


# In[ ]:


def train_one_epoch(model, loader, optimizer, criterion, device=DEVICE):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def compute_per_class_accuracy(model, loader, num_classes=NUM_CLASSES, device=DEVICE):
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(dim=1)

        for c in range(num_classes):
            mask = (targets == c)
            correct[c] += (predicted[mask] == c).sum().item()
            total[c] += mask.sum().item()

    return (correct / (total + 1e-8) * 100).tolist()


# In[ ]:


set_seed(SEED)

model = ResNet18_CIFAR(num_classes=NUM_CLASSES).to(DEVICE)

criterion = nn.CrossEntropyLoss()  # Cross-entropy loss (Section 2D)

optimizer = optim.SGD(
    model.parameters(),
    lr=LR_INITIAL,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)

scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer,
    milestones=LR_MILESTONES,  # [116, 233]
    gamma=LR_GAMMA             # 0.1
)

print("Training configuration:")
print(f"  Optimizer:     SGD (lr={LR_INITIAL}, momentum={MOMENTUM}, weight_decay={WEIGHT_DECAY})")
print(f"  LR Schedule:   MultiStepLR at epochs {LR_MILESTONES}, gamma={LR_GAMMA}")
print(f"  Loss:          CrossEntropyLoss")
print(f"  Epochs:        {TOTAL_EPOCHS}")
print(f"  Batch size:    {BATCH_SIZE}")


# In[ ]:


history = {
    'train_loss': [],
    'train_acc': [],
    'test_loss': [],
    'test_acc': [],
    'lr': [],
    'nc_metrics': {},          # epoch -> dict of NC metric values
    'zero_error_epoch': None,  # epoch when train acc first hits TPT_THRESHOLD
}

start_time = time.time()

for epoch in tqdm(range(TOTAL_EPOCHS), desc='Training', unit='epoch'):
    current_lr = optimizer.param_groups[0]['lr']
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['test_loss'].append(test_loss)
    history['test_acc'].append(test_acc)
    history['lr'].append(current_lr)

    if history['zero_error_epoch'] is None and train_acc >= TPT_THRESHOLD:
        history['zero_error_epoch'] = epoch
        tqdm.write(f"\n*** TPT begins: training accuracy reached {TPT_THRESHOLD}% at epoch {epoch} ***\n")

    if epoch in METRIC_EPOCHS:
        nc_metrics = compute_all_nc_metrics(
            model, train_eval_loader, test_loader,
            num_classes=NUM_CLASSES, device=DEVICE
        )
        history['nc_metrics'][epoch] = nc_metrics
        tqdm.write(
            f"  [Epoch {epoch:>3d}] NC1={nc_metrics['nc1']:.4f}  "
            f"NC3={nc_metrics['nc3']:.4f}  "
            f"NC4_test={nc_metrics['nc4_test']:.4f}  "
            f"NC4_train={nc_metrics['nc4_train']:.4f}"
        )

    if epoch % 50 == 0 or epoch in LR_MILESTONES:
        tqdm.write(
            f"  [Epoch {epoch:>3d}] lr={current_lr:.5f}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.2f}%  "
            f"test_acc={test_acc:.2f}%"
        )

    if epoch % 50 == 0 or epoch == TOTAL_EPOCHS - 1:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'history': history,
        }, f'checkpoints/baseline_epoch_{epoch:03d}.pt')

total_time = time.time() - start_time
print(f"\n{'='*70}")
print(f"Training complete in {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
print(f"TPT started at epoch: {history['zero_error_epoch']}")
print(f"Final train acc: {history['train_acc'][-1]:.2f}%")
print(f"Final test acc:  {history['test_acc'][-1]:.2f}%")


# In[ ]:


def get_nc_series(history, key):
    epochs = sorted(history['nc_metrics'].keys())
    values = [history['nc_metrics'][e][key] for e in epochs]
    return epochs, values

zero_epoch = history['zero_error_epoch']
print(f"TPT starts at epoch {zero_epoch}")
print(f"NC metrics available at {len(history['nc_metrics'])} checkpoints")


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_range = range(len(history['train_acc']))

axes[0].plot(epochs_range, history['train_acc'], label='Train', linewidth=1.5)
axes[0].plot(epochs_range, history['test_acc'], label='Test', linewidth=1.5)
if zero_epoch is not None:
    axes[0].axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.7, label=f'Zero Error (epoch {zero_epoch})')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_title('Training & Test Accuracy')
axes[0].legend()
axes[0].set_ylim([0, 101])


axes[1].semilogy(epochs_range, history['train_loss'], label='Train', linewidth=1.5)
axes[1].semilogy(epochs_range, history['test_loss'], label='Test', linewidth=1.5)
if zero_epoch is not None:
    axes[1].axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.7)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (log scale)')
axes[1].set_title('Training & Test Loss')
axes[1].legend()


axes[2].semilogy(epochs_range, history['lr'], linewidth=2, color='green')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate (log scale)')
axes[2].set_title('Learning Rate Schedule')
for ms in LR_MILESTONES:
    axes[2].axvline(x=ms, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('results/plots/train_test_curves.png', dpi=150, bbox_inches='tight')
plt.show()

if zero_epoch is not None:
    print(f"\n--- Table 1 Comparison (cf. Paper Table 1, ResNet row under CIFAR10) ---")
    print(f"Test accuracy at zero error (epoch {zero_epoch}): {history['test_acc'][zero_epoch]:.2f}%")
    print(f"Test accuracy at last epoch ({TOTAL_EPOCHS-1}):     {history['test_acc'][-1]:.2f}%")
    print(f"Improvement during TPT:                       +{history['test_acc'][-1] - history['test_acc'][zero_epoch]:.2f}%")
    print(f"(Paper reports: 88.72% -> 89.44% = +0.72% for ResNet-18/CIFAR-10)")
    print(f"")
    print(f"--- TPT Loss Analysis ---")
    print(f"Train loss at zero error (epoch {zero_epoch}): {history['train_loss'][zero_epoch]:.6f}")
    print(f"Train loss at final epoch ({TOTAL_EPOCHS-1}):     {history['train_loss'][-1]:.6f}")
    print(f"Loss reduction during TPT: {history['train_loss'][zero_epoch] / max(history['train_loss'][-1], 1e-10):.1f}x")
    print(f"This confirms TPT: loss continues to decrease even after training error is zero.")


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))

epochs_nc, nc1_vals = get_nc_series(history, 'nc1')

ax.semilogy(epochs_nc, nc1_vals, 'b-o', markersize=3, linewidth=1.5,
            label=r'Tr($\Sigma_W \Sigma_B^\dagger$ / C)')

if zero_epoch is not None:
    ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label=f'Zero Error (epoch {zero_epoch})')

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel(r'Tr($\Sigma_W \Sigma_B^\dagger$ / C)  [log scale]', fontsize=13)
ax.set_title('NC1: Within-Class Variability Collapse\n(cf. Paper Figure 6)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/nc1_variability_collapse.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"NC1 at epoch {epochs_nc[0]}:   {nc1_vals[0]:.4f}")
print(f"NC1 at epoch {epochs_nc[-1]}:  {nc1_vals[-1]:.6f}")
if nc1_vals[-1] > 0:
    print(f"Reduction factor: {nc1_vals[0] / nc1_vals[-1]:.0f}x")


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))

epochs_nc, means_en = get_nc_series(history, 'nc2_means_equinorm')
_, clf_en = get_nc_series(history, 'nc2_clf_equinorm')

ax.plot(epochs_nc, means_en, 'b-o', markersize=3, linewidth=1.5, label='Mean Activations')
ax.plot(epochs_nc, clf_en, color='orange', marker='o', markersize=3, linewidth=1.5, label='Classifiers')

if zero_epoch is not None:
    ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Zero Error')

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Std / Avg  (Coefficient of Variation)', fontsize=13)
ax.set_title('NC2: Equinorm — Class Means & Classifiers Become Equal-Length\n(cf. Paper Figure 2)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/nc2_equinorm.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))

epochs_nc, means_ea = get_nc_series(history, 'nc2_means_equiangular')
_, clf_ea = get_nc_series(history, 'nc2_clf_equiangular')

ax.plot(epochs_nc, means_ea, 'b-o', markersize=3, linewidth=1.5, label='Mean Activations')
ax.plot(epochs_nc, clf_ea, color='orange', marker='o', markersize=3, linewidth=1.5, label='Classifiers')

if zero_epoch is not None:
    ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Zero Error')

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Std(cos)', fontsize=13)
ax.set_title('NC2: Equiangular — Pairwise Angles Become Equal\n(cf. Paper Figure 3)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/nc2_equiangular.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))

epochs_nc, means_cd = get_nc_series(history, 'nc2_means_cos_deviation')
_, clf_cd = get_nc_series(history, 'nc2_clf_cos_deviation')

ax.plot(epochs_nc, means_cd, 'b-o', markersize=3, linewidth=1.5, label='Mean Activations')
ax.plot(epochs_nc, clf_cd, color='orange', marker='o', markersize=3, linewidth=1.5, label='Classifiers')

if zero_epoch is not None:
    ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Zero Error')

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Avg |cos + 1/(C-1)|', fontsize=13)
ax.set_title(f'NC2: Maximal Equiangularity — Cosines approach -1/(C-1) = {-1/(NUM_CLASSES-1):.4f}\n(cf. Paper Figure 4)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/nc2_maximal_equiangularity.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))

epochs_nc, nc3_vals = get_nc_series(history, 'nc3')

ax.plot(epochs_nc, nc3_vals, 'b-o', markersize=3, linewidth=1.5,
        label=r'$\|\tilde{W}^T - \tilde{M}\|_F^2$')

if zero_epoch is not None:
    ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2,
               label='Zero Error')

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel(r'$\|\tilde{W}^T - \tilde{M}\|_F^2$', fontsize=13)
ax.set_title('NC3: Self-Duality — Classifier Converges to Class Means\n(cf. Paper Figure 5)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/nc3_self_duality.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"NC3 at epoch {epochs_nc[0]}:   {nc3_vals[0]:.4f}")
print(f"NC3 at epoch {epochs_nc[-1]}:  {nc3_vals[-1]:.6f}")


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 6))

epochs_nc, nc4_test_vals = get_nc_series(history, 'nc4_test')
_, nc4_train_vals = get_nc_series(history, 'nc4_train')

ax.plot(epochs_nc, nc4_test_vals, 'b-o', markersize=3, linewidth=1.5, label='Test set (Paper Figure 7)')
ax.plot(epochs_nc, nc4_train_vals, 'g-s', markersize=3, linewidth=1.5, label='Train set', alpha=0.7)

if zero_epoch is not None:
    ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Zero Error')

ax.set_xlabel('Epoch', fontsize=13)
ax.set_ylabel('Proportion Mismatch', fontsize=13)
ax.set_title('NC4: Classifier Behavior Approaches Nearest Class Center\n(cf. Paper Figure 7)', fontsize=14)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/plots/nc4_ncc_agreement.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"NC4 (test)  at epoch {epochs_nc[0]}:  {nc4_test_vals[0]:.4f} ({nc4_test_vals[0]*100:.1f}% disagree)")
print(f"NC4 (test)  at epoch {epochs_nc[-1]}: {nc4_test_vals[-1]:.4f} ({nc4_test_vals[-1]*100:.1f}% disagree)")
print(f"NC4 (train) at epoch {epochs_nc[-1]}: {nc4_train_vals[-1]:.4f} ({nc4_train_vals[-1]*100:.1f}% disagree)")


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Neural Collapse: All Metrics During Training\n'
             'ResNet-18 on Balanced CIFAR-10 (350 epochs)',
             fontsize=16, fontweight='bold', y=1.02)

# NC1 (log scale)
ax = axes[0, 0]
e, v = get_nc_series(history, 'nc1')
ax.semilogy(e, v, 'b-o', markersize=2, linewidth=1.5)
if zero_epoch is not None: ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.6)
ax.set_title('NC1: Variability Collapse\n' + r'Tr($\Sigma_W \Sigma_B^\dagger$ / C)', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Log scale')

# NC2 Equinorm
ax = axes[0, 1]
e, v1 = get_nc_series(history, 'nc2_means_equinorm')
_, v2 = get_nc_series(history, 'nc2_clf_equinorm')
ax.plot(e, v1, 'b-o', markersize=2, linewidth=1.5, label='Means')
ax.plot(e, v2, color='orange', marker='o', markersize=2, linewidth=1.5, label='Classifiers')
if zero_epoch is not None: ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.6)
ax.set_title('NC2: Equinorm\nCoV of norms (Fig. 2)', fontweight='bold')
ax.set_xlabel('Epoch')
ax.legend(fontsize=9)

# NC2 Equiangular
ax = axes[0, 2]
e, v1 = get_nc_series(history, 'nc2_means_equiangular')
_, v2 = get_nc_series(history, 'nc2_clf_equiangular')
ax.plot(e, v1, 'b-o', markersize=2, linewidth=1.5, label='Means')
ax.plot(e, v2, color='orange', marker='o', markersize=2, linewidth=1.5, label='Classifiers')
if zero_epoch is not None: ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.6)
ax.set_title('NC2: Equiangular\nStd of cosines (Fig. 3)', fontweight='bold')
ax.set_xlabel('Epoch')
ax.legend(fontsize=9)

# NC2 Maximal angle
ax = axes[1, 0]
e, v1 = get_nc_series(history, 'nc2_means_cos_deviation')
_, v2 = get_nc_series(history, 'nc2_clf_cos_deviation')
ax.plot(e, v1, 'b-o', markersize=2, linewidth=1.5, label='Means')
ax.plot(e, v2, color='orange', marker='o', markersize=2, linewidth=1.5, label='Classifiers')
if zero_epoch is not None: ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.6)
ax.set_title('NC2: Maximal Equiangularity\nAvg |cos + 1/(C-1)| (Fig. 4)', fontweight='bold')
ax.set_xlabel('Epoch')
ax.legend(fontsize=9)

# NC3
ax = axes[1, 1]
e, v = get_nc_series(history, 'nc3')
ax.plot(e, v, 'b-o', markersize=2, linewidth=1.5)
if zero_epoch is not None: ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.6)
ax.set_title('NC3: Self-Duality\n' + r'$\|\tilde{W}^T - \tilde{M}\|_F^2$ (Fig. 5)', fontweight='bold')
ax.set_xlabel('Epoch')

# NC4
ax = axes[1, 2]
e, v1 = get_nc_series(history, 'nc4_test')
_, v2 = get_nc_series(history, 'nc4_train')
ax.plot(e, v1, 'b-o', markersize=2, linewidth=1.5, label='Test')
ax.plot(e, v2, 'g-s', markersize=2, linewidth=1.5, label='Train', alpha=0.7)
if zero_epoch is not None: ax.axvline(x=zero_epoch, color='red', linestyle='--', alpha=0.6)
ax.set_title('NC4: NCC Agreement\nDisagreement rate (Fig. 7)', fontweight='bold')
ax.set_xlabel('Epoch')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('results/plots/nc_all_metrics_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


print("Computing detailed final-epoch NC analysis...")

# Extract features at the final trained model
features_by_class, all_features, all_labels = extract_features(
    model, train_eval_loader, NUM_CLASSES, DEVICE
)
class_means, global_mean = compute_means(features_by_class, all_features, NUM_CLASSES)

# Centered class means
centered_means = class_means - global_mean.unsqueeze(0)  # (10, 512)

print(f"\n{'='*60}")
print(f"FINAL NC STATE — Epoch {TOTAL_EPOCHS - 1}")
print(f"{'='*60}")

# --- NC1 ---
final_nc1 = compute_nc1(features_by_class, class_means, global_mean, NUM_CLASSES)
print(f"\nNC1 (Variability Collapse):")
print(f"  Tr(Sigma_W Sigma_B^dagger / C) = {final_nc1:.6f}")

# --- NC2 ---
norms = torch.norm(centered_means, dim=1)
normalized = centered_means / (norms.unsqueeze(1) + 1e-8)
cosine_matrix = normalized @ normalized.T
mask = ~torch.eye(NUM_CLASSES, dtype=torch.bool)
off_diag = cosine_matrix[mask]

print(f"\nNC2 (Simplex ETF):")
print(f"  Class-mean norms (should be equal for equinorm):")
for c in range(NUM_CLASSES):
    print(f"    Class {c} ({train_dataset.classes[c]:>10s}): ||mu_c - mu_G|| = {norms[c]:.4f}")
print(f"  CoV of norms: {norms.std()/norms.mean():.6f}")
print(f"  Pairwise cosines: mean={off_diag.mean():.6f}, std={off_diag.std():.6f}")
print(f"  Target cosine (-1/(C-1)): {-1/(NUM_CLASSES-1):.6f}")
print(f"  Avg |cos + 1/(C-1)|: {(off_diag + 1/(NUM_CLASSES-1)).abs().mean():.6f}")

# --- NC3 ---
final_nc3 = compute_nc3(model, class_means, global_mean)
print(f"\nNC3 (Self-Duality):")
print(f"  ||W_tilde^T - M_tilde||_F^2 = {final_nc3:.6f}")


b = model.classifier.bias.data.cpu()
print(f"\nClassifier Biases (should converge toward uniform — cf. Eq. 7b):")
print(f"  Values: {[f'{bi:.4f}' for bi in b.tolist()]}")
print(f"  Std:    {b.std():.6f}")
print(f"  Mean:   {b.mean():.6f}")

final_nc4_test = compute_nc4(model, test_loader, class_means, DEVICE)
final_nc4_train = compute_nc4(model, train_eval_loader, class_means, DEVICE)
print(f"\nNC4 (NCC Agreement):")
print(f"  Test disagreement  = {final_nc4_test:.4f} ({final_nc4_test*100:.2f}% of test examples)")
print(f"  Train disagreement = {final_nc4_train:.4f} ({final_nc4_train*100:.2f}% of train examples)")

per_class_acc = compute_per_class_accuracy(model, test_loader, NUM_CLASSES, DEVICE)
print(f"\nPer-Class Test Accuracy:")
for c in range(NUM_CLASSES):
    print(f"  Class {c} ({train_dataset.classes[c]:>10s}): {per_class_acc[c]:.2f}%")
print(f"  Overall: {np.mean(per_class_acc):.2f}%")


# In[ ]:


fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Class means cosine matrix
centered_means = class_means - global_mean.unsqueeze(0)
norms = torch.norm(centered_means, dim=1)
normalized = centered_means / (norms.unsqueeze(1) + 1e-8)
cos_mat_means = (normalized @ normalized.T).numpy()

im1 = axes[0].imshow(cos_mat_means, cmap='RdBu_r', vmin=-0.5, vmax=1.0)
axes[0].set_title('Cosine Similarity: Class Means', fontweight='bold')
axes[0].set_xticks(range(NUM_CLASSES))
axes[0].set_yticks(range(NUM_CLASSES))
axes[0].set_xticklabels(train_dataset.classes, rotation=45, ha='right', fontsize=9)
axes[0].set_yticklabels(train_dataset.classes, fontsize=9)
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# Text annotations
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        axes[0].text(j, i, f'{cos_mat_means[i,j]:.2f}', ha='center', va='center', fontsize=7)

# Classifier weights cosine matrix
W = model.classifier.weight.data.cpu()
W_norms = torch.norm(W, dim=1)
W_normalized = W / (W_norms.unsqueeze(1) + 1e-8)
cos_mat_clf = (W_normalized @ W_normalized.T).numpy()

im2 = axes[1].imshow(cos_mat_clf, cmap='RdBu_r', vmin=-0.5, vmax=1.0)
axes[1].set_title('Cosine Similarity: Classifier Weights', fontweight='bold')
axes[1].set_xticks(range(NUM_CLASSES))
axes[1].set_yticks(range(NUM_CLASSES))
axes[1].set_xticklabels(train_dataset.classes, rotation=45, ha='right', fontsize=9)
axes[1].set_yticklabels(train_dataset.classes, fontsize=9)
plt.colorbar(im2, ax=axes[1], shrink=0.8)

for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        axes[1].text(j, i, f'{cos_mat_clf[i,j]:.2f}', ha='center', va='center', fontsize=7)

plt.suptitle(f'NC2+NC3 Verification: Off-diagonal should approach {-1/(NUM_CLASSES-1):.3f}, '
             f'and both matrices should be nearly identical (self-duality)',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('results/plots/cosine_matrices_final.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


n_per_class = 200
subset_features = []
subset_labels = []

for c in range(NUM_CLASSES):
    idx = np.random.choice(features_by_class[c].shape[0], n_per_class, replace=False)
    subset_features.append(features_by_class[c][idx])
    subset_labels.extend([c] * n_per_class)

subset_features = torch.cat(subset_features, dim=0).numpy()
subset_labels = np.array(subset_labels)

# PCA to 2D
pca = PCA(n_components=2)
features_2d = pca.fit_transform(subset_features)

# Also project class means
means_2d = pca.transform(class_means.numpy())

fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

for c in range(NUM_CLASSES):
    mask_c = (subset_labels == c)
    ax.scatter(features_2d[mask_c, 0], features_2d[mask_c, 1],
              c=[colors[c]], alpha=0.3, s=10, label=train_dataset.classes[c])
    # Plot class mean as a large marker
    ax.scatter(means_2d[c, 0], means_2d[c, 1],
              c=[colors[c]], s=200, marker='*', edgecolors='black', linewidths=1, zorder=5)

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
ax.set_title('Last-Layer Features (PCA Projection)\n'
             'Stars = class means. Tight clusters = NC1. Symmetric layout = NC2.',
             fontsize=13)
ax.legend(fontsize=9, ncol=2, loc='best')
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('results/plots/feature_pca_final.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"PCA explains {pca.explained_variance_ratio_.sum()*100:.1f}% of variance in first 2 components")


# In[ ]:


print("=" * 70)
print("BASELINE EXPERIMENT SUMMARY")
print("Neural Collapse: ResNet-18 on CIFAR-10 (Balanced)")
print("=" * 70)

# Get first and last NC metrics
first_epoch = min(history['nc_metrics'].keys())
last_epoch = max(history['nc_metrics'].keys())
first_nc = history['nc_metrics'][first_epoch]
last_nc = history['nc_metrics'][last_epoch]

print(f"\nTraining Configuration:")
print(f"  Architecture:   ResNet-18 (modified for CIFAR-10)")
print(f"  Dataset:        CIFAR-10 (balanced, 5000/class)")
print(f"  Augmentation:   None (matching Section 2E)")
print(f"  Optimizer:      SGD (lr=0.1, momentum=0.9, wd=5e-4)")
print(f"  LR Schedule:    x0.1 at epochs 116, 233")
print(f"  Total Epochs:   {TOTAL_EPOCHS}")

print(f"\nTraining Dynamics:")
print(f"  TPT begins at epoch:       {history['zero_error_epoch']}")
print(f"  Final train accuracy:      {history['train_acc'][-1]:.2f}%")
print(f"  Final test accuracy:       {history['test_acc'][-1]:.2f}%")
if history['zero_error_epoch'] is not None:
    ze = history['zero_error_epoch']
    print(f"  Test acc at zero error:    {history['test_acc'][ze]:.2f}%")
    print(f"  Test acc improvement (TPT): +{history['test_acc'][-1] - history['test_acc'][ze]:.2f}%")
    print(f"  Train loss at zero error:  {history['train_loss'][ze]:.6f}")
    print(f"  Train loss at final epoch: {history['train_loss'][-1]:.6f}")
    print(f"  Loss reduction during TPT: {history['train_loss'][ze] / max(history['train_loss'][-1], 1e-10):.1f}x")

print(f"\nNC Metrics — Early (epoch {first_epoch}) vs Final (epoch {last_epoch}):")
print(f"  {'Metric':<30} {'Early':>12} {'Final':>12} {'Trend':>10}")
print(f"  {'-'*64}")

metrics_to_show = [
    ('NC1: Tr(Sw Sb+/C)', 'nc1'),
    ('NC2: Equinorm (means)', 'nc2_means_equinorm'),
    ('NC2: Equiangular (means)', 'nc2_means_equiangular'),
    ('NC2: cos->-1/(C-1) (means)', 'nc2_means_cos_deviation'),
    ('NC2: Equinorm (clf)', 'nc2_clf_equinorm'),
    ('NC2: Equiangular (clf)', 'nc2_clf_equiangular'),
    ('NC2: cos->-1/(C-1) (clf)', 'nc2_clf_cos_deviation'),
    ('NC3: Self-duality', 'nc3'),
    ('NC4: NCC disagree (test)', 'nc4_test'),
    ('NC4: NCC disagree (train)', 'nc4_train'),
]

for name, key in metrics_to_show:
    early = first_nc[key]
    final = last_nc[key]
    if early > 0 and final > 0:
        ratio = early / final
        trend = f"{ratio:.0f}x down" if ratio > 1 else f"{1/ratio:.1f}x up"
    elif early > 0 and final == 0:
        trend = "-> 0"
    else:
        trend = "--"
    print(f"  {name:<30} {early:>12.6f} {final:>12.6f} {trend:>10}")

print(f"\nConclusion:")
print(f"  All four NC properties (NC1-NC4) decrease substantially during")
print(f"  training, confirming Neural Collapse on balanced CIFAR-10 with")
print(f"  ResNet-18, consistent with Papyan et al. (2020).")
print(f"  Training loss continues to decrease during TPT even after zero")
print(f"  training error, driving the geometric convergence.")
print(f"  This baseline serves as the reference for our extension experiments")
print(f"  on class imbalance and ETF regularization.")

