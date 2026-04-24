# Self-Pruning Neural Network (CIFAR-10)

## Overview

This project implements a neural network that **learns to prune its own weights during training** using learnable gates.

Each weight is multiplied by a gate:

* `gate = sigmoid(gate_scores)`
* If gate → 0 → weight is effectively removed

---

## Model

* Fully connected network: 3072 → 512 → 256 → 10
* Activation: ReLU
* Custom layer: `PrunableLinear`

---

## Loss Function

Total Loss = CrossEntropy + λ × Sparsity Loss

* CrossEntropy → accuracy
* Sparsity Loss → encourages pruning

---

## Training

* Dataset: CIFAR-10
* Optimizer: Adam
* Epochs: 25
* Tested λ values: 0.0001, 0.001, 0.01

---

## Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 0.0001 | 56.6%    | 22.9%    |
| 0.001  | 52.0%    | 93.9%    |
| 0.01   | 43.3%    | 99.8%    |

---

## Observation

As λ increases:

* Sparsity increases
* Accuracy decreases

This shows the expected **trade-off between model compression and performance**.

---

## Run

```
pip install -r requirements.txt
python main.py
```
