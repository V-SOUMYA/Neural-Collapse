# Neural Collapse in Deep Learning

---

## Overview

This project investigates **Neural Collapse (NC)**, a phenomenon identified by Papyan, Han, and Donoho (2020) in deep neural classifiers.

Near the end of training, the internal representations learned by a network converge to a highly structured geometric configuration:

- **NC1:** Features from the same class collapse toward a single class mean.
- **NC2:** Class means arrange themselves into a symmetric simplex Equiangular Tight Frame (ETF).
- **NC3:** Final-layer classifier weights align with the class means.
- **NC4:** Classification becomes equivalent to a nearest-class-mean decision rule.

While the original work demonstrates these properties under standard training conditions, this project explores how Neural Collapse behaves when those assumptions are relaxed, particularly under **class imbalance**.

---

## Project Goals

### 1. Reproduce Neural Collapse on CIFAR-10

Train a ResNet-18 on CIFAR-10 and track all four Neural Collapse metrics throughout training.

Metrics include:

- NC1: Within-class vs. between-class covariance ratio
- NC2: ETF alignment of class means
- NC3: Alignment between classifier weights and class means
- NC4: Agreement with nearest-class-mean classification

This serves as the baseline experiment.

---

### 2. Stress-Test Neural Collapse Under Class Imbalance

Create long-tail versions of CIFAR-10 with varying imbalance levels and analyze how Neural Collapse properties degrade.

Key questions:

- Which NC properties break first?
- Does imbalance affect feature geometry more than classifier alignment?
- How closely do our observations match the findings of Fang et al. (2021)?

---

### 3. ETF-Based Regularization

Rather than only observing Neural Collapse, we explore whether it can be encouraged directly.

A regularization term will be added to encourage class means to follow the ETF geometry associated with NC2.

We investigate:

- Whether ETF regularization improves performance under imbalance.
- Whether any gains arise from improved feature geometry or from a general regularization effect.

---

### 4. Linear Probing as a Representation Diagnostic

At selected checkpoints:

1. Freeze the learned feature extractor.
2. Train a linear classifier on top of the frozen features.
3. Compare linear probe accuracy with Neural Collapse metrics.

This helps determine whether stronger Neural Collapse corresponds to genuinely better representations.

---

### 5. Neural Collapse and Generalization

Track:

- Overall test accuracy
- Per-class accuracy
- Per-class NC metrics

The goal is to understand whether stronger collapse predicts better generalization and whether degradation of NC under imbalance correlates with poor performance on minority classes.

---

## Dataset

### CIFAR-10

- 10 image classes
- 50,000 training images
- 10,000 test images
- Balanced class distribution

Long-tail variants will be generated to simulate varying levels of class imbalance.

Dataset: https://www.cs.toronto.edu/~kriz/cifar.html

---

## References

### Primary Paper

Papyan, Han, and Donoho (2020)

**Prevalence of Neural Collapse During the Terminal Phase of Deep Learning Training**

https://www.pnas.org/doi/10.1073/pnas.2015509117

### Extension Paper

Fang et al. (2021)

**Explore the Neural Collapse Phenomenon in Long-Tail Learning**

https://arxiv.org/abs/2110.02180


---

## Planned Experiments

| Experiment | Goal |
|------------|------|
| Baseline CIFAR-10 Training | Reproduce NC1–NC4 |
| Long-Tail CIFAR-10 | Study NC under class imbalance |
| ETF Regularization | Encourage NC2 geometry during training |
| Linear Probing | Measure representation quality |
| Generalization Analysis | Relate NC metrics to test performance |

---

## Expected Outcomes

This project aims to better understand:

- Whether Neural Collapse remains robust under severe class imbalance.
- Which collapse properties are most sensitive to data distribution shifts.
- Whether explicitly encouraging ETF geometry can improve performance.
- How closely Neural Collapse metrics track representation quality and generalization.
