#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

import os
import random
import time
import copy
import warnings
warnings.filterwarnings('ignore')

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

SEED = 42

def set_seed(seed=SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")


# In[ ]:


NUM_CLASSES   = 10
FEATURE_DIM   = 512
BATCH_SIZE    = 128
TOTAL_EPOCHS  = 350
LR_INITIAL    = 0.1
MOMENTUM      = 0.9
WEIGHT_DECAY  = 5e-4
LR_MILESTONES = [TOTAL_EPOCHS // 3, 2 * TOTAL_EPOCHS // 3]  # [116, 233]
LR_GAMMA      = 0.1
TPT_THRESHOLD = 99.9

METRIC_EPOCHS = sorted(set(
    list(range(0, TOTAL_EPOCHS, 10)) +
    list(range(TOTAL_EPOCHS - 50, TOTAL_EPOCHS))
))

IMBALANCE_RATIOS = [1, 10, 50, 100]

os.makedirs('results/imbalance', exist_ok=True)
os.makedirs('results/imbalance/plots', exist_ok=True)
os.makedirs('checkpoints/imbalance', exist_ok=True)

print(f"Will train {len(IMBALANCE_RATIOS)} models with imbalance ratios: {IMBALANCE_RATIOS}")
print(f"Each model trains for {TOTAL_EPOCHS} epochs")


# In[ ]:


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

full_train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

test_loader = DataLoader(
    test_dataset, batch_size=256, shuffle=False,
    num_workers=2, pin_memory=True
)

CLASS_NAMES = full_train_dataset.classes
print(f"Full training set: {len(full_train_dataset)} images")
print(f"Test set: {len(test_dataset)} images")
print(f"Classes: {CLASS_NAMES}")


# In[ ]:


def create_imbalanced_subset(dataset, imbalance_ratio, num_classes=10, max_per_class=5000, seed=42):
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)
    samples_per_class = []
    for c in range(num_classes):
        n_c = int(max_per_class * (imbalance_ratio ** (-c / (num_classes - 1))))
        n_c = max(n_c, 2)  # at least 2 samples for covariance computation
        samples_per_class.append(n_c)
    
    all_indices = []
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        chosen = rng.choice(class_indices, size=samples_per_class[c], replace=False)
        all_indices.extend(chosen.tolist())
    
    subset = Subset(dataset, all_indices)
    return subset, samples_per_class


print(f"{'Ratio':>6s}  {'Total':>6s}  ", end="")
for c in range(NUM_CLASSES):
    print(f"C{c:d}", end="    ")
print()
print("-" * 80)

for rho in IMBALANCE_RATIOS:
    _, spc = create_imbalanced_subset(full_train_dataset, rho)
    total = sum(spc)
    print(f"rho={rho:<4d}  {total:>6d}  ", end="")
    for c in range(NUM_CLASSES):
        print(f"{spc[c]:>5d}", end=" ")
    print()


# In[ ]:


class ResNet18_CIFAR(nn.Module):
    def __init__(self, num_classes=10, feature_dim=512):
        super().__init__()
        resnet = models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,
        )
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x, return_features=False):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        logits = self.classifier(h)
        if return_features:
            return logits, h
        return logits


# In[ ]:


@torch.no_grad()
def extract_features(model, loader, num_classes=NUM_CLASSES, device=DEVICE):
    model.eval()
    all_features, all_labels = [], []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        _, h = model(inputs, return_features=True)
        all_features.append(h.cpu())
        all_labels.append(targets)
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    features_by_class = {}
    for c in range(num_classes):
        mask = (all_labels == c)
        features_by_class[c] = all_features[mask]
    return features_by_class, all_features, all_labels


def compute_means(features_by_class, all_features, num_classes=NUM_CLASSES):
    class_means = torch.stack([
        features_by_class[c].mean(dim=0) for c in range(num_classes)
    ], dim=0)
    global_mean = all_features.mean(dim=0)
    return class_means, global_mean


def compute_nc1(features_by_class, class_means, global_mean, num_classes=NUM_CLASSES):
    feature_dim = class_means.shape[1]
    Sigma_W = torch.zeros(feature_dim, feature_dim, dtype=torch.float64)
    total_count = 0
    for c in range(num_classes):
        centered = (features_by_class[c] - class_means[c].unsqueeze(0)).double()
        Sigma_W += centered.T @ centered
        total_count += features_by_class[c].shape[0]
    Sigma_W /= total_count
    centered_means = (class_means - global_mean.unsqueeze(0)).double()
    Sigma_B = (centered_means.T @ centered_means) / num_classes
    Sigma_B_pinv = torch.from_numpy(np.linalg.pinv(Sigma_B.numpy(), rcond=1e-10))
    nc1 = torch.trace(Sigma_W @ Sigma_B_pinv).item() / num_classes
    return nc1


def compute_nc1_per_class(features_by_class, class_means, num_classes=NUM_CLASSES):
    per_class_var = []
    for c in range(num_classes):
        centered = features_by_class[c] - class_means[c].unsqueeze(0)
        avg_sq_dist = (centered ** 2).sum(dim=1).mean().item()
        per_class_var.append(avg_sq_dist)
    return per_class_var


def compute_nc2(class_means, global_mean, num_classes=NUM_CLASSES):
    centered_means = class_means - global_mean.unsqueeze(0)
    norms = torch.norm(centered_means, dim=1)
    equinorm_cv = (norms.std() / (norms.mean() + 1e-8)).item()
    normalized = centered_means / (norms.unsqueeze(1) + 1e-8)
    cosine_matrix = normalized @ normalized.T
    mask = ~torch.eye(num_classes, dtype=torch.bool)
    off_diag = cosine_matrix[mask]
    equiangular_std = off_diag.std().item()
    target = -1.0 / (num_classes - 1)
    cos_deviation = (off_diag - target).abs().mean().item()
    return equinorm_cv, equiangular_std, cos_deviation


def compute_nc2_classifier(model, num_classes=NUM_CLASSES):
    W = model.classifier.weight.data.cpu()
    norms = torch.norm(W, dim=1)
    equinorm_cv = (norms.std() / (norms.mean() + 1e-8)).item()
    normalized = W / (norms.unsqueeze(1) + 1e-8)
    cosine_matrix = normalized @ normalized.T
    mask = ~torch.eye(num_classes, dtype=torch.bool)
    off_diag = cosine_matrix[mask]
    equiangular_std = off_diag.std().item()
    target = -1.0 / (num_classes - 1)
    cos_deviation = (off_diag - target).abs().mean().item()
    return equinorm_cv, equiangular_std, cos_deviation


def compute_nc3(model, class_means, global_mean):
    W = model.classifier.weight.data.cpu().float()
    M_dot = (class_means - global_mean.unsqueeze(0)).T.float()
    W_norm = W / (torch.norm(W, p='fro') + 1e-8)
    M_norm = M_dot / (torch.norm(M_dot, p='fro') + 1e-8)
    nc3 = (torch.norm(W_norm.T - M_norm, p='fro') ** 2).item()
    return nc3


@torch.no_grad()
def compute_nc4(model, loader, class_means, device=DEVICE):
    model.eval()
    class_means_dev = class_means.to(device)
    total, disagreements = 0, 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        logits, features = model(inputs, return_features=True)
        net_pred = logits.argmax(dim=1)
        dists = torch.cdist(features, class_means_dev, p=2)
        ncc_pred = dists.argmin(dim=1)
        disagreements += (net_pred != ncc_pred).sum().item()
        total += inputs.size(0)
    return disagreements / total


def compute_all_nc_metrics(model, train_eval_loader, test_loader,
                           num_classes=NUM_CLASSES, device=DEVICE):
    features_by_class, all_features, all_labels = extract_features(
        model, train_eval_loader, num_classes, device
    )
    class_means, global_mean = compute_means(features_by_class, all_features, num_classes)
    
    nc1 = compute_nc1(features_by_class, class_means, global_mean, num_classes)
    nc1_per_class = compute_nc1_per_class(features_by_class, class_means, num_classes)
    means_equinorm, means_equiangular, means_cos_dev = compute_nc2(class_means, global_mean, num_classes)
    clf_equinorm, clf_equiangular, clf_cos_dev = compute_nc2_classifier(model, num_classes)
    nc3 = compute_nc3(model, class_means, global_mean)
    nc4_test = compute_nc4(model, test_loader, class_means, device)
    nc4_train = compute_nc4(model, train_eval_loader, class_means, device)
    W = model.classifier.weight.data.cpu()
    clf_norms_per_class = torch.norm(W, dim=1).tolist()
    centered_means = class_means - global_mean.unsqueeze(0)
    mean_norms_per_class = torch.norm(centered_means, dim=1).tolist()
    
    return {
        'nc1': nc1,
        'nc1_per_class': nc1_per_class,
        'nc2_means_equinorm': means_equinorm,
        'nc2_means_equiangular': means_equiangular,
        'nc2_means_cos_deviation': means_cos_dev,
        'nc2_clf_equinorm': clf_equinorm,
        'nc2_clf_equiangular': clf_equiangular,
        'nc2_clf_cos_deviation': clf_cos_dev,
        'nc3': nc3,
        'nc4_test': nc4_test,
        'nc4_train': nc4_train,
        'clf_norms_per_class': clf_norms_per_class,
        'mean_norms_per_class': mean_norms_per_class,
    }


# In[ ]:


def train_one_epoch(model, loader, optimizer, criterion, device=DEVICE):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
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
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device=DEVICE):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def compute_per_class_accuracy(model, loader, num_classes=NUM_CLASSES, device=DEVICE):
    model.eval()
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        _, predicted = model(inputs).max(dim=1)
        for c in range(num_classes):
            mask = (targets == c)
            correct[c] += (predicted[mask] == c).sum().item()
            total[c] += mask.sum().item()
    return (correct / (total + 1e-8) * 100).tolist()


# In[ ]:


all_results = {}  # rho -> history dict

for rho in IMBALANCE_RATIOS:
    print(f"\n{'='*70}")
    print(f"  IMBALANCE RATIO rho = {rho}")
    print(f"{'='*70}")
    
    set_seed(SEED)
    train_subset, samples_per_class = create_imbalanced_subset(
        full_train_dataset, rho, seed=SEED
    )
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True
    )
    train_eval_loader = DataLoader(
        train_subset, batch_size=256, shuffle=False,
        num_workers=2, pin_memory=True
    )
    
    total_samples = sum(samples_per_class)
    print(f"  Training samples: {total_samples}")
    print(f"  Samples/class: {samples_per_class}")
    print(f"  Max/Min ratio: {max(samples_per_class)}/{min(samples_per_class)} = {max(samples_per_class)/min(samples_per_class):.1f}")
    model = ResNet18_CIFAR(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR_INITIAL,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                          milestones=LR_MILESTONES, gamma=LR_GAMMA)
    

    history = {
        'rho': rho,
        'samples_per_class': samples_per_class,
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'lr': [],
        'nc_metrics': {},
        'zero_error_epoch': None,
    }
    
    start_time = time.time()
    
    for epoch in tqdm(range(TOTAL_EPOCHS), desc=f'rho={rho}', unit='epoch', leave=True):
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
            tqdm.write(f"  *** TPT at epoch {epoch} ***")
        
        if epoch in METRIC_EPOCHS:
            nc = compute_all_nc_metrics(model, train_eval_loader, test_loader)
            history['nc_metrics'][epoch] = nc
        
        if epoch % 100 == 0 or epoch == TOTAL_EPOCHS - 1:
            tqdm.write(f"  [E{epoch:>3d}] loss={train_loss:.4f} train={train_acc:.1f}% test={test_acc:.1f}%")
    
    elapsed = time.time() - start_time
    history['final_per_class_acc'] = compute_per_class_accuracy(model, test_loader)
    print(f"  Finished in {elapsed/60:.0f} min | TPT: {history['zero_error_epoch']} | "
          f"Final test: {history['test_acc'][-1]:.2f}%")
    
    all_results[rho] = history
    torch.save({
        'history': history,
        'model_state_dict': model.state_dict(),
    }, f'checkpoints/imbalance/rho_{rho}.pt')

# Save everything
torch.save(all_results, 'results/imbalance/all_results.pt')
print(f"\nAll {len(IMBALANCE_RATIOS)} experiments complete. Saved to results/imbalance/")


# In[ ]:


def get_nc_series(history, key):
    epochs = sorted(history['nc_metrics'].keys())
    values = [history['nc_metrics'][e][key] for e in epochs]
    return epochs, values

# Color scheme for imbalance ratios
RATIO_COLORS = {1: 'tab:blue', 10: 'tab:orange', 50: 'tab:red', 100: 'darkred'}
RATIO_LABELS = {1: 'ρ=1 (balanced)', 10: 'ρ=10', 50: 'ρ=50', 100: 'ρ=100'}


# In[ ]:


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for rho, hist in all_results.items():
    c = RATIO_COLORS[rho]
    lbl = RATIO_LABELS[rho]
    epochs_range = range(len(hist['train_acc']))
    axes[0].plot(epochs_range, hist['train_acc'], color=c, linewidth=1.2, label=lbl)
    axes[1].plot(epochs_range, hist['test_acc'], color=c, linewidth=1.2, label=lbl)
    axes[2].semilogy(epochs_range, hist['train_loss'], color=c, linewidth=1.2, label=lbl)

axes[0].set_title('Train Accuracy')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy (%)')
axes[0].legend(fontsize=10); axes[0].set_ylim([0, 101])

axes[1].set_title('Test Accuracy')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
axes[1].legend(fontsize=10)

axes[2].set_title('Train Loss')
axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Loss (log scale)')
axes[2].legend(fontsize=10)

plt.suptitle('Training Dynamics vs. Imbalance Ratio', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/imbalance/plots/training_dynamics.png', dpi=150, bbox_inches='tight')
plt.show()

# Summary table
print(f"{'Ratio':>6s}  {'TPT Epoch':>10s}  {'Final Train':>12s}  {'Final Test':>12s}")
print("-" * 50)
for rho, hist in all_results.items():
    ze = hist['zero_error_epoch']
    ze_str = str(ze) if ze is not None else "Never"
    print(f"rho={rho:<4d}  {ze_str:>10s}  {hist['train_acc'][-1]:>11.2f}%  {hist['test_acc'][-1]:>11.2f}%")


# In[ ]:


fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Neural Collapse Metrics vs. Imbalance Ratio', fontsize=16, fontweight='bold', y=1.02)

metric_configs = [
    (axes[0,0], 'nc1', 'NC1: Variability Collapse\nTr(Σ_W Σ_B† / C)', True),
    (axes[0,1], 'nc2_means_equinorm', 'NC2a: Equinorm (means)\nCoV of ‖μ_c − μ_G‖', False),
    (axes[0,2], 'nc2_means_cos_deviation', 'NC2c: Maximal Equiangularity\nAvg |cos + 1/(C-1)|', False),
    (axes[1,0], 'nc3', 'NC3: Self-Duality\n‖W̃ᵀ − M̃‖²_F', False),
    (axes[1,1], 'nc4_test', 'NC4: NCC Disagreement (test)\nProportion mismatch', False),
    (axes[1,2], 'nc2_clf_equinorm', 'NC2a: Equinorm (classifier)\nCoV of ‖w_c‖', False),
]

for ax, key, title, use_log in metric_configs:
    for rho, hist in all_results.items():
        e, v = get_nc_series(hist, key)
        if use_log:
            ax.semilogy(e, v, color=RATIO_COLORS[rho], linewidth=1.5,
                       marker='o', markersize=1.5, label=RATIO_LABELS[rho])
        else:
            ax.plot(e, v, color=RATIO_COLORS[rho], linewidth=1.5,
                   marker='o', markersize=1.5, label=RATIO_LABELS[rho])
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/imbalance/plots/nc_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


fig, axes = plt.subplots(1, len(IMBALANCE_RATIOS), figsize=(5*len(IMBALANCE_RATIOS), 5))
if len(IMBALANCE_RATIOS) == 1:
    axes = [axes]

for i, rho in enumerate(IMBALANCE_RATIOS):
    ax = axes[i]
    hist = all_results[rho]
    last_epoch = max(hist['nc_metrics'].keys())
    nc_final = hist['nc_metrics'][last_epoch]
    
    spc = hist['samples_per_class']
    clf_norms = nc_final['clf_norms_per_class']
    
    ax.bar(range(NUM_CLASSES), clf_norms, color=[RATIO_COLORS[rho]]*NUM_CLASSES, alpha=0.7)
    ax.set_xlabel('Class')
    ax.set_ylabel('‖w_c‖')
    ax.set_title(f'ρ={rho}', fontweight='bold')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f'{c}\n({spc[c]})' for c in range(NUM_CLASSES)], fontsize=8)

plt.suptitle('Classifier Norms per Class (end of training)\n'
             'Fang et al. predict: larger class → larger ‖w_c‖',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/imbalance/plots/classifier_norms_vs_frequency.png', dpi=150, bbox_inches='tight')
plt.show()
print("Correlation between class sample count and classifier norm:")
for rho in IMBALANCE_RATIOS:
    hist = all_results[rho]
    last_epoch = max(hist['nc_metrics'].keys())
    spc = np.array(hist['samples_per_class'])
    norms = np.array(hist['nc_metrics'][last_epoch]['clf_norms_per_class'])
    corr = np.corrcoef(spc, norms)[0, 1]
    print(f"  rho={rho:<4d}: r = {corr:+.4f}")


# In[ ]:


fig, axes = plt.subplots(1, len(IMBALANCE_RATIOS), figsize=(5*len(IMBALANCE_RATIOS), 5))
if len(IMBALANCE_RATIOS) == 1:
    axes = [axes]

for i, rho in enumerate(IMBALANCE_RATIOS):
    ax = axes[i]
    hist = all_results[rho]
    spc = hist['samples_per_class']
    pca_vals = hist['final_per_class_acc']
    
    colors_bar = [plt.cm.RdYlGn(acc / 100.0) for acc in pca_vals]
    ax.bar(range(NUM_CLASSES), pca_vals, color=colors_bar, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Class')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'ρ={rho} (avg={np.mean(pca_vals):.1f}%)', fontweight='bold')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f'{c}\n({spc[c]})' for c in range(NUM_CLASSES)], fontsize=8)
    ax.set_ylim([0, 105])

plt.suptitle('Per-Class Test Accuracy\n'
             'Numbers below class labels show training sample count',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/imbalance/plots/per_class_accuracy.png', dpi=150, bbox_inches='tight')
plt.show()

# Correlation
print("Correlation between class sample count and test accuracy:")
for rho in IMBALANCE_RATIOS:
    hist = all_results[rho]
    spc = np.array(hist['samples_per_class'])
    acc = np.array(hist['final_per_class_acc'])
    corr = np.corrcoef(spc, acc)[0, 1]
    print(f"  rho={rho:<4d}: r = {corr:+.4f}  minority acc = {acc[-1]:.1f}%  majority acc = {acc[0]:.1f}%")


# In[ ]:


fig, axes = plt.subplots(1, len(IMBALANCE_RATIOS), figsize=(5*len(IMBALANCE_RATIOS), 5))
if len(IMBALANCE_RATIOS) == 1:
    axes = [axes]

for i, rho in enumerate(IMBALANCE_RATIOS):
    ax = axes[i]
    hist = all_results[rho]
    last_epoch = max(hist['nc_metrics'].keys())
    nc1_pc = hist['nc_metrics'][last_epoch]['nc1_per_class']
    spc = hist['samples_per_class']
    
    ax.bar(range(NUM_CLASSES), nc1_pc, color=[RATIO_COLORS[rho]]*NUM_CLASSES, alpha=0.7)
    ax.set_xlabel('Class')
    ax.set_ylabel('Avg ‖h - μ_c‖²')
    ax.set_title(f'ρ={rho}', fontweight='bold')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_xticklabels([f'{c}\n({spc[c]})' for c in range(NUM_CLASSES)], fontsize=8)

plt.suptitle('Per-Class Within-Class Variance (NC1)\n'
             'Do minority classes have looser clusters?',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/imbalance/plots/nc1_per_class.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


fig, axes = plt.subplots(1, len(IMBALANCE_RATIOS), figsize=(5*len(IMBALANCE_RATIOS), 4.5))
if len(IMBALANCE_RATIOS) == 1:
    axes = [axes]

for i, rho in enumerate(IMBALANCE_RATIOS):
    ax = axes[i]
    hist = all_results[rho]
    last_epoch = max(hist['nc_metrics'].keys())
    nc = hist['nc_metrics'][last_epoch]
    ckpt = torch.load(f'checkpoints/imbalance/rho_{rho}.pt', weights_only=False, map_location='cpu')
    temp_model = ResNet18_CIFAR(num_classes=NUM_CLASSES)
    temp_model.load_state_dict(ckpt['model_state_dict'])
    temp_model.eval()
    W = temp_model.classifier.weight.data
    W_norms = torch.norm(W, dim=1)
    W_normalized = W / (W_norms.unsqueeze(1) + 1e-8)
    cos_mat = (W_normalized @ W_normalized.T).numpy()
    im = ax.imshow(cos_mat, cmap='RdBu_r', vmin=-0.5, vmax=1.0)
    ax.set_title(f'ρ={rho}', fontweight='bold')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    spc = hist['samples_per_class']
    ax.set_xticklabels([f'{spc[c]}' for c in range(NUM_CLASSES)], fontsize=7, rotation=45)
    ax.set_yticklabels([f'C{c}' for c in range(NUM_CLASSES)], fontsize=7)
    
    for ii in range(NUM_CLASSES):
        for jj in range(NUM_CLASSES):
            ax.text(jj, ii, f'{cos_mat[ii,jj]:.1f}', ha='center', va='center', fontsize=5)

plt.suptitle('Classifier Weight Cosine Matrices\n'
             'Balanced: uniform off-diagonal ≈ -0.11  |  Imbalanced: symmetry breaks',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('results/imbalance/plots/cosine_matrices.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


print(f"{'':>30s}", end="")
for rho in IMBALANCE_RATIOS:
    print(f"  {'rho='+str(rho):>12s}", end="")
print()
print("-" * (30 + 14 * len(IMBALANCE_RATIOS)))

metrics_to_compare = [
    ('NC1: Tr(Sw Sb+/C)', 'nc1'),
    ('NC2: Equinorm (means)', 'nc2_means_equinorm'),
    ('NC2: Equiangular (means)', 'nc2_means_equiangular'),
    ('NC2: MaxAngle (means)', 'nc2_means_cos_deviation'),
    ('NC2: Equinorm (clf)', 'nc2_clf_equinorm'),
    ('NC3: Self-duality', 'nc3'),
    ('NC4: NCC disagree (test)', 'nc4_test'),
]

for name, key in metrics_to_compare:
    print(f"  {name:<28s}", end="")
    for rho in IMBALANCE_RATIOS:
        hist = all_results[rho]
        last_e = max(hist['nc_metrics'].keys())
        val = hist['nc_metrics'][last_e][key]
        print(f"  {val:>12.6f}", end="")
    print()

print()
print(f"{'Test Accuracy':<30s}", end="")
for rho in IMBALANCE_RATIOS:
    print(f"  {all_results[rho]['test_acc'][-1]:>11.2f}%", end="")
print()

print(f"{'Minority (C9) Accuracy':<30s}", end="")
for rho in IMBALANCE_RATIOS:
    print(f"  {all_results[rho]['final_per_class_acc'][9]:>11.2f}%", end="")
print()

