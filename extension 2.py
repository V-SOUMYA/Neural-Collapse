#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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

# NC metric + linear probe checkpoints
METRIC_EPOCHS = sorted(set(
    list(range(0, TOTAL_EPOCHS, 10)) +
    list(range(TOTAL_EPOCHS - 50, TOTAL_EPOCHS))
))

# Linear probe checkpoints (less frequent — it's expensive)
PROBE_EPOCHS = sorted(set(
    list(range(0, TOTAL_EPOCHS, 25)) +
    [TOTAL_EPOCHS - 1]
))

IMBALANCE_RATIOS = [10, 50, 100]
ETF_LAMBDAS = [0.0, 0.01, 0.1, 1.0]

os.makedirs('results/etf_reg', exist_ok=True)
os.makedirs('results/etf_reg/plots', exist_ok=True)
os.makedirs('checkpoints/etf_reg', exist_ok=True)

print(f"Imbalance ratios: {IMBALANCE_RATIOS}")
print(f"ETF lambda values: {ETF_LAMBDAS}")
print(f"Total experiments: {len(IMBALANCE_RATIOS) * len(ETF_LAMBDAS)}")


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
print(f"Training set: {len(full_train_dataset)} images")
print(f"Test set: {len(test_dataset)} images")


def create_imbalanced_subset(dataset, imbalance_ratio, num_classes=10,
                             max_per_class=5000, seed=42):
    rng = np.random.RandomState(seed)
    targets = np.array(dataset.targets)

    samples_per_class = []
    for c in range(num_classes):
        n_c = int(max_per_class * (imbalance_ratio ** (-c / (num_classes - 1))))
        n_c = max(n_c, 2)
        samples_per_class.append(n_c)

    all_indices = []
    for c in range(num_classes):
        class_indices = np.where(targets == c)[0]
        chosen = rng.choice(class_indices, size=samples_per_class[c], replace=False)
        all_indices.extend(chosen.tolist())

    subset = Subset(dataset, all_indices)
    return subset, samples_per_class


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


def compute_etf_target(num_classes):
    C = num_classes
    G_star = (C / (C - 1)) * (
        torch.eye(C) - (1.0 / C) * torch.ones(C, C)
    )
    return G_star


def etf_regularization_loss(classifier_weight, etf_target):
    W = classifier_weight  # (C, p)
    W_hat = W / (torch.norm(W, p='fro') + 1e-8)  # (C, p)
    G_actual = W_hat @ W_hat.T  # (C, C)
    loss = torch.norm(G_actual - etf_target.to(W.device), p='fro') ** 2
    return loss

G_star = compute_etf_target(NUM_CLASSES)
print(f"ETF target Gram matrix (C={NUM_CLASSES}):")
print(f"  Diagonal:     {G_star[0, 0]:.4f} (should be 1.0)")
print(f"  Off-diagonal: {G_star[0, 1]:.4f} (should be {-1/(NUM_CLASSES-1):.4f})")
print(f"  Frobenius norm: {torch.norm(G_star, p='fro'):.4f}")


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


def linear_probe(model, train_loader, test_loader, num_classes=NUM_CLASSES,
                 device=DEVICE, max_iter=1000):
    # Extract training features
    features_by_class, train_features, train_labels = extract_features(
        model, train_loader, num_classes, device
    )
    
    # Extract test features
    _, test_features, test_labels = extract_features(
        model, test_loader, num_classes, device
    )
    
    X_train = train_features.numpy()
    y_train = train_labels.numpy()
    X_test = test_features.numpy()
    y_test = test_labels.numpy()
    
    # Standardize features (important for logistic regression convergence)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Fit multinomial logistic regression
    # Using lbfgs solver which handles multiclass well
    clf = LogisticRegression(
        max_iter=max_iter,
        solver='lbfgs',
        multi_class='multinomial',
        C=1.0,  # inverse regularization strength
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    
    # Overall accuracy
    overall_acc = clf.score(X_test, y_test) * 100.0
    
    # Per-class accuracy
    y_pred = clf.predict(X_test)
    per_class_acc = []
    for c in range(num_classes):
        mask = (y_test == c)
        if mask.sum() > 0:
            acc = (y_pred[mask] == c).mean() * 100.0
        else:
            acc = 0.0
        per_class_acc.append(acc)
    
    return overall_acc, per_class_acc


# In[ ]:


def train_one_epoch_etf(model, loader, optimizer, criterion,
                        etf_target, etf_lambda, device=DEVICE):

    model.train()
    running_loss = 0.0
    running_ce_loss = 0.0
    running_etf_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # Cross-entropy loss
        ce_loss = criterion(outputs, targets)
        
        # ETF regularization loss
        if etf_lambda > 0:
            etf_loss = etf_regularization_loss(
                model.classifier.weight, etf_target
            )
            total_loss = ce_loss + etf_lambda * etf_loss
        else:
            etf_loss = torch.tensor(0.0)
            total_loss = ce_loss
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item() * inputs.size(0)
        running_ce_loss += ce_loss.item() * inputs.size(0)
        running_etf_loss += etf_loss.item() * inputs.size(0)
        _, predicted = outputs.max(dim=1)
        correct += predicted.eq(targets).sum().item()
        total += targets.size(0)
    
    return (running_loss / total, running_ce_loss / total,
            running_etf_loss / total, 100.0 * correct / total)


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


ETF_TARGET = compute_etf_target(NUM_CLASSES)

all_results = {}  # (rho, lambda) -> history dict

for rho in IMBALANCE_RATIOS:
    for lam in ETF_LAMBDAS:
        exp_key = (rho, lam)
        print(f"\n{'='*70}")
        print(f"  IMBALANCE ρ={rho}, ETF λ={lam}")
        print(f"{'='*70}")
        
        set_seed(SEED)
        
        # --- Create imbalanced dataset ---
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
        
        # --- Initialize model ---
        model = ResNet18_CIFAR(num_classes=NUM_CLASSES).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=LR_INITIAL,
                              momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA
        )
        
        # --- Training history ---
        history = {
            'rho': rho,
            'etf_lambda': lam,
            'samples_per_class': samples_per_class,
            'train_loss': [], 'train_ce_loss': [], 'train_etf_loss': [],
            'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'lr': [],
            'nc_metrics': {},
            'linear_probe': {},  # epoch -> {overall_acc, per_class_acc}
            'zero_error_epoch': None,
        }
        
        start_time = time.time()
        
        for epoch in tqdm(range(TOTAL_EPOCHS), desc=f'ρ={rho},λ={lam}',
                          unit='epoch', leave=True):
            current_lr = optimizer.param_groups[0]['lr']
            
            # --- Train one epoch with ETF regularization ---
            total_loss, ce_loss, etf_loss, train_acc = train_one_epoch_etf(
                model, train_loader, optimizer, criterion,
                ETF_TARGET, lam
            )
            
            # --- Evaluate on test set ---
            test_loss, test_acc = evaluate(model, test_loader, criterion)
            
            scheduler.step()
            
            # --- Record history ---
            history['train_loss'].append(total_loss)
            history['train_ce_loss'].append(ce_loss)
            history['train_etf_loss'].append(etf_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['lr'].append(current_lr)
            
            # --- Track TPT ---
            if history['zero_error_epoch'] is None and train_acc >= TPT_THRESHOLD:
                history['zero_error_epoch'] = epoch
                tqdm.write(f"  *** TPT at epoch {epoch} ***")
            
            # --- Compute NC metrics ---
            if epoch in METRIC_EPOCHS:
                nc = compute_all_nc_metrics(model, train_eval_loader, test_loader)
                history['nc_metrics'][epoch] = nc
            
            # --- Linear probing ---
            if epoch in PROBE_EPOCHS:
                probe_acc, probe_per_class = linear_probe(
                    model, train_eval_loader, test_loader
                )
                history['linear_probe'][epoch] = {
                    'overall_acc': probe_acc,
                    'per_class_acc': probe_per_class,
                }
                tqdm.write(f"  [E{epoch:>3d}] probe={probe_acc:.1f}% test={test_acc:.1f}%")
            
            # --- Print progress ---
            if epoch % 100 == 0 or epoch == TOTAL_EPOCHS - 1:
                tqdm.write(
                    f"  [E{epoch:>3d}] ce={ce_loss:.4f} etf={etf_loss:.4f} "
                    f"train={train_acc:.1f}% test={test_acc:.1f}%"
                )
        
        elapsed = time.time() - start_time
        
        # --- Final per-class accuracy ---
        history['final_per_class_acc'] = compute_per_class_accuracy(model, test_loader)
        
        print(f"  Finished in {elapsed/60:.0f} min | TPT: {history['zero_error_epoch']} | "
              f"Final test: {history['test_acc'][-1]:.2f}%")
        
        all_results[exp_key] = history
        
        # Save per-experiment checkpoint
        torch.save({
            'history': history,
            'model_state_dict': model.state_dict(),
        }, f'checkpoints/etf_reg/rho_{rho}_lam_{lam}.pt')

# Save all results
torch.save(all_results, 'results/etf_reg/all_results.pt')
print(f"\nAll {len(all_results)} experiments complete.")


# In[ ]:


def get_nc_series(history, key):
    epochs = sorted(history['nc_metrics'].keys())
    values = [history['nc_metrics'][e][key] for e in epochs]
    return epochs, values

def get_probe_series(history):
    epochs = sorted(history['linear_probe'].keys())
    values = [history['linear_probe'][e]['overall_acc'] for e in epochs]
    return epochs, values

# Color scheme for lambda values
LAM_COLORS = {0.0: 'tab:blue', 0.01: 'tab:green', 0.1: 'tab:orange', 1.0: 'tab:red'}
LAM_LABELS = {0.0: 'λ=0 (baseline)', 0.01: 'λ=0.01', 0.1: 'λ=0.1', 1.0: 'λ=1.0'}


# In[ ]:


fig, axes = plt.subplots(len(IMBALANCE_RATIOS), 3, 
                         figsize=(18, 5*len(IMBALANCE_RATIOS)))
if len(IMBALANCE_RATIOS) == 1:
    axes = axes.reshape(1, -1)

for row, rho in enumerate(IMBALANCE_RATIOS):
    for lam in ETF_LAMBDAS:
        hist = all_results[(rho, lam)]
        c = LAM_COLORS[lam]
        lbl = LAM_LABELS[lam]
        
        # NC2 Equinorm (means)
        e, v = get_nc_series(hist, 'nc2_means_equinorm')
        axes[row, 0].plot(e, v, color=c, linewidth=1.5, label=lbl)
        
        # NC2 Equiangular Std (means)
        e, v = get_nc_series(hist, 'nc2_means_equiangular')
        axes[row, 1].plot(e, v, color=c, linewidth=1.5, label=lbl)
        
        # NC2 Max angle deviation (means)
        e, v = get_nc_series(hist, 'nc2_means_cos_deviation')
        axes[row, 2].plot(e, v, color=c, linewidth=1.5, label=lbl)
    
    axes[row, 0].set_title(f'ρ={rho}: Equinorm (CoV)', fontweight='bold')
    axes[row, 0].set_ylabel('Std/Avg of ‖μ_c − μ_G‖')
    axes[row, 1].set_title(f'ρ={rho}: Equiangularity (Std cos)', fontweight='bold')
    axes[row, 2].set_title(f'ρ={rho}: Max Angle (|cos + 1/(C-1)|)', fontweight='bold')
    
    for j in range(3):
        axes[row, j].set_xlabel('Epoch')
        axes[row, j].legend(fontsize=8)
        axes[row, j].grid(True, alpha=0.3)

plt.suptitle('Effect of ETF Regularization on NC2 (Class Mean Geometry)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/etf_reg/plots/nc2_vs_lambda.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


fig, axes = plt.subplots(len(IMBALANCE_RATIOS), 3,
                         figsize=(18, 5*len(IMBALANCE_RATIOS)))
if len(IMBALANCE_RATIOS) == 1:
    axes = axes.reshape(1, -1)

for row, rho in enumerate(IMBALANCE_RATIOS):
    for lam in ETF_LAMBDAS:
        hist = all_results[(rho, lam)]
        c = LAM_COLORS[lam]
        lbl = LAM_LABELS[lam]
        
        # NC1
        e, v = get_nc_series(hist, 'nc1')
        axes[row, 0].semilogy(e, v, color=c, linewidth=1.5, label=lbl)
        
        # NC3
        e, v = get_nc_series(hist, 'nc3')
        axes[row, 1].plot(e, v, color=c, linewidth=1.5, label=lbl)
        
        # NC4
        e, v = get_nc_series(hist, 'nc4_test')
        axes[row, 2].plot(e, v, color=c, linewidth=1.5, label=lbl)
    
    axes[row, 0].set_title(f'ρ={rho}: NC1 (Variability Collapse)', fontweight='bold')
    axes[row, 1].set_title(f'ρ={rho}: NC3 (Self-Duality)', fontweight='bold')
    axes[row, 2].set_title(f'ρ={rho}: NC4 (NCC Disagreement)', fontweight='bold')
    
    for j in range(3):
        axes[row, j].set_xlabel('Epoch')
        axes[row, j].legend(fontsize=8)
        axes[row, j].grid(True, alpha=0.3)

plt.suptitle('Effect of ETF Regularization on NC1, NC3, NC4',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/etf_reg/plots/nc134_vs_lambda.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


fig, axes = plt.subplots(1, len(IMBALANCE_RATIOS),
                         figsize=(6*len(IMBALANCE_RATIOS), 6))
if len(IMBALANCE_RATIOS) == 1:
    axes = [axes]

for i, rho in enumerate(IMBALANCE_RATIOS):
    ax = axes[i]
    
    x = np.arange(NUM_CLASSES)
    width = 0.8 / len(ETF_LAMBDAS)
    
    for j, lam in enumerate(ETF_LAMBDAS):
        hist = all_results[(rho, lam)]
        pca_vals = hist['final_per_class_acc']
        offset = (j - len(ETF_LAMBDAS)/2 + 0.5) * width
        bars = ax.bar(x + offset, pca_vals, width, label=LAM_LABELS[lam],
                      color=LAM_COLORS[lam], alpha=0.7, edgecolor='black',
                      linewidth=0.3)
    
    spc = all_results[(rho, 0.0)]['samples_per_class']
    ax.set_xlabel('Class (samples)')
    ax.set_ylabel('Test Accuracy (%)')
    ax.set_title(f'ρ={rho}', fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n({spc[c]})' for c in range(NUM_CLASSES)], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim([0, 105])

plt.suptitle('Per-Class Test Accuracy: ETF Regularization vs Baseline\n'
             'Does forcing ETF geometry help minority classes?',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/etf_reg/plots/per_class_acc_vs_lambda.png', dpi=150, bbox_inches='tight')
plt.show()

# Print minority class improvement summary
print("\nMinority class (C9) accuracy across experiments:")
print(f"{'rho':>5s}", end="")
for lam in ETF_LAMBDAS:
    print(f"  {'λ='+str(lam):>12s}", end="")
print()
print("-" * (5 + 14 * len(ETF_LAMBDAS)))
for rho in IMBALANCE_RATIOS:
    print(f"{rho:>5d}", end="")
    for lam in ETF_LAMBDAS:
        acc = all_results[(rho, lam)]['final_per_class_acc'][9]
        print(f"  {acc:>11.2f}%", end="")
    print()


# In[ ]:


fig, axes = plt.subplots(1, len(IMBALANCE_RATIOS),
                         figsize=(6*len(IMBALANCE_RATIOS), 5))
if len(IMBALANCE_RATIOS) == 1:
    axes = [axes]

for i, rho in enumerate(IMBALANCE_RATIOS):
    ax = axes[i]
    for lam in ETF_LAMBDAS:
        hist = all_results[(rho, lam)]
        pe, pv = get_probe_series(hist)
        ax.plot(pe, pv, color=LAM_COLORS[lam], linewidth=2,
                marker='o', markersize=4, label=LAM_LABELS[lam])
        
        # Also plot test accuracy (dashed) for comparison
        ax.plot(range(len(hist['test_acc'])), hist['test_acc'],
                color=LAM_COLORS[lam], linewidth=0.8, alpha=0.4, linestyle='--')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'ρ={rho}: Linear Probe (solid) vs Test (dashed)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Linear Probe Accuracy Over Training\n'
             'Probe measures representation quality independent of classifier bias',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/etf_reg/plots/linear_probe_over_training.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 15b: Scatter plot — NC1 vs. linear probe accuracy ---
fig, axes = plt.subplots(1, len(IMBALANCE_RATIOS),
                         figsize=(6*len(IMBALANCE_RATIOS), 5))
if len(IMBALANCE_RATIOS) == 1:
    axes = [axes]

for i, rho in enumerate(IMBALANCE_RATIOS):
    ax = axes[i]
    for lam in ETF_LAMBDAS:
        hist = all_results[(rho, lam)]
        
        # Get NC1 and probe values at probe epochs
        probe_epochs = sorted(hist['linear_probe'].keys())
        nc1_at_probe = []
        probe_accs = []
        for ep in probe_epochs:
            if ep in hist['nc_metrics']:
                nc1_at_probe.append(hist['nc_metrics'][ep]['nc1'])
                probe_accs.append(hist['linear_probe'][ep]['overall_acc'])
        
        if len(nc1_at_probe) > 0:
            ax.scatter(nc1_at_probe, probe_accs, color=LAM_COLORS[lam],
                      s=40, alpha=0.7, label=LAM_LABELS[lam], edgecolors='black',
                      linewidths=0.5)
    
    ax.set_xlabel('NC1: Tr(Σ_W Σ_B† / C)')
    ax.set_ylabel('Linear Probe Accuracy (%)')
    ax.set_title(f'ρ={rho}: NC1 vs. Probe Quality', fontweight='bold')
    ax.set_xscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Does Neural Collapse Predict Representation Quality?\n'
             'Each point is one training checkpoint. Lower NC1 → features collapsed more.',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/etf_reg/plots/nc1_vs_probe.png', dpi=150, bbox_inches='tight')
plt.show()

# --- 15c: Final probe per-class accuracy for minority analysis ---
fig, axes = plt.subplots(1, len(IMBALANCE_RATIOS),
                         figsize=(6*len(IMBALANCE_RATIOS), 6))
if len(IMBALANCE_RATIOS) == 1:
    axes = [axes]

for i, rho in enumerate(IMBALANCE_RATIOS):
    ax = axes[i]
    x = np.arange(NUM_CLASSES)
    width = 0.8 / len(ETF_LAMBDAS)
    
    for j, lam in enumerate(ETF_LAMBDAS):
        hist = all_results[(rho, lam)]
        last_probe_epoch = max(hist['linear_probe'].keys())
        probe_pc = hist['linear_probe'][last_probe_epoch]['per_class_acc']
        offset = (j - len(ETF_LAMBDAS)/2 + 0.5) * width
        ax.bar(x + offset, probe_pc, width, label=LAM_LABELS[lam],
               color=LAM_COLORS[lam], alpha=0.7, edgecolor='black', linewidth=0.3)
    
    spc = all_results[(rho, 0.0)]['samples_per_class']
    ax.set_xlabel('Class (samples)')
    ax.set_ylabel('Linear Probe Accuracy (%)')
    ax.set_title(f'ρ={rho}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}\n({spc[c]})' for c in range(NUM_CLASSES)], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim([0, 105])

plt.suptitle('Per-Class Linear Probe Accuracy at Final Epoch\n'
             'Probe bypasses classifier bias — measures pure feature quality per class',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/etf_reg/plots/probe_per_class_final.png', dpi=150, bbox_inches='tight')
plt.show()


# In[ ]:


print("COMPREHENSIVE SUMMARY: ETF REGULARIZATION EXPERIMENTS")
for rho in IMBALANCE_RATIOS:
    print(f"\n--- Imbalance ρ = {rho} ---")
    spc = all_results[(rho, 0.0)]['samples_per_class']
    print(f"  Samples/class: {spc}")
    
    print(f"\n  {'Metric':<35s}", end="")
    for lam in ETF_LAMBDAS:
        print(f"  {'λ='+str(lam):>10s}", end="")
    print()
    print("  " + "-"*75)
    
    metrics_table = [
        ('Test Accuracy (%)', lambda h: h['test_acc'][-1]),
        ('Minority (C9) Acc (%)', lambda h: h['final_per_class_acc'][9]),
        ('Majority (C0) Acc (%)', lambda h: h['final_per_class_acc'][0]),
        ('NC1: Tr(Sw Sb+/C)', lambda h: h['nc_metrics'][max(h['nc_metrics'].keys())]['nc1']),
        ('NC2: Equinorm (means)', lambda h: h['nc_metrics'][max(h['nc_metrics'].keys())]['nc2_means_equinorm']),
        ('NC2: Cos deviation', lambda h: h['nc_metrics'][max(h['nc_metrics'].keys())]['nc2_means_cos_deviation']),
        ('NC3: Self-duality', lambda h: h['nc_metrics'][max(h['nc_metrics'].keys())]['nc3']),
        ('NC4: NCC disagree', lambda h: h['nc_metrics'][max(h['nc_metrics'].keys())]['nc4_test']),
        ('Linear Probe (%)', lambda h: h['linear_probe'][max(h['linear_probe'].keys())]['overall_acc']),
        ('Probe Minority (C9) (%)', lambda h: h['linear_probe'][max(h['linear_probe'].keys())]['per_class_acc'][9]),
    ]
    
    for name, fn in metrics_table:
        print(f"  {name:<35s}", end="")
        for lam in ETF_LAMBDAS:
            val = fn(all_results[(rho, lam)])
            print(f"  {val:>10.4f}", end="")
        print()


# In[ ]:


print("DECOMPOSITION: Feature Effect vs Classifier Effect")

for rho in IMBALANCE_RATIOS:
    print(f"\n--- ρ = {rho} ---")
    print(f"  {'λ':>6s}  {'Test%':>8s}  {'Probe%':>8s}  {'Gap':>8s}  {'NC2 cos':>10s}")
    print("  " + "-"*48)
    
    for lam in ETF_LAMBDAS:
        hist = all_results[(rho, lam)]
        test_acc = hist['test_acc'][-1]
        last_probe = max(hist['linear_probe'].keys())
        probe_acc = hist['linear_probe'][last_probe]['overall_acc']
        gap = probe_acc - test_acc
        last_nc = max(hist['nc_metrics'].keys())
        nc2_cos = hist['nc_metrics'][last_nc]['nc2_means_cos_deviation']
        
        print(f"  {lam:>6.2f}  {test_acc:>7.2f}%  {probe_acc:>7.2f}%  "
              f"{gap:>+7.2f}%  {nc2_cos:>10.6f}")
    
    print()
    print("  Interpretation:")
    baseline = all_results[(rho, 0.0)]
    base_test = baseline['test_acc'][-1]
    base_probe = baseline['linear_probe'][max(baseline['linear_probe'].keys())]['overall_acc']
    
    best_lam = max(ETF_LAMBDAS[1:],
                   key=lambda l: all_results[(rho, l)]['test_acc'][-1])
    best = all_results[(rho, best_lam)]
    best_test = best['test_acc'][-1]
    best_probe = best['linear_probe'][max(best['linear_probe'].keys())]['overall_acc']
    
    test_delta = best_test - base_test
    probe_delta = best_probe - base_probe
    
    if probe_delta > 0.5:
        print(f"  → ETF reg (λ={best_lam}) improved probe by {probe_delta:+.2f}%")
        print(f"    This suggests the regularization improved feature geometry directly.")
    elif test_delta > 0.5 and abs(probe_delta) <= 0.5:
        print(f"  → ETF reg (λ={best_lam}) improved test by {test_delta:+.2f}% but not probe")
        print(f"    This suggests the benefit is through the classifier, not features.")
    elif test_delta < -0.5:
        print(f"  → ETF reg (λ={best_lam}) hurt test by {test_delta:+.2f}%")
        print(f"    The geometric constraint may be too strong for this imbalance level.")
    else:
        print(f"  → ETF reg had minimal effect (test delta: {test_delta:+.2f}%)")


# In[ ]:


for rho in IMBALANCE_RATIOS:
    fig, axes = plt.subplots(1, len(ETF_LAMBDAS),
                             figsize=(5*len(ETF_LAMBDAS), 4.5))
    
    for i, lam in enumerate(ETF_LAMBDAS):
        ax = axes[i]
        
        ckpt = torch.load(f'checkpoints/etf_reg/rho_{rho}_lam_{lam}.pt',
                          weights_only=False, map_location='cpu')
        temp_model = ResNet18_CIFAR(num_classes=NUM_CLASSES)
        temp_model.load_state_dict(ckpt['model_state_dict'])
        
        W = temp_model.classifier.weight.data
        W_norms = torch.norm(W, dim=1)
        W_normalized = W / (W_norms.unsqueeze(1) + 1e-8)
        cos_mat = (W_normalized @ W_normalized.T).numpy()
        
        im = ax.imshow(cos_mat, cmap='RdBu_r', vmin=-0.5, vmax=1.0)
        ax.set_title(f'λ={lam}', fontweight='bold')
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        spc = all_results[(rho, lam)]['samples_per_class']
        ax.set_xticklabels([f'{spc[c]}' for c in range(NUM_CLASSES)],
                           fontsize=7, rotation=45)
        ax.set_yticklabels([f'C{c}' for c in range(NUM_CLASSES)], fontsize=7)
        
        for ii in range(NUM_CLASSES):
            for jj in range(NUM_CLASSES):
                ax.text(jj, ii, f'{cos_mat[ii,jj]:.1f}',
                       ha='center', va='center', fontsize=5)
    
    plt.suptitle(f'Classifier Cosine Matrices (ρ={rho})\n'
                 f'Target: diag=1, off-diag={-1/(NUM_CLASSES-1):.3f}. '
                 f'Does ETF reg restore symmetry?',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'results/etf_reg/plots/cosine_matrices_rho_{rho}.png',
               dpi=150, bbox_inches='tight')
    plt.show()

