# Self-Pruning Neural Network (CIFAR-10)

## Overview

This project implements a neural network that **learns to prune its own weights during training** using learnable gates.

Each weight is controlled by a gate:

* `gate = sigmoid(gate_scores)`
* If gate → 0 → weight is effectively removed

---

## Model

* Fully connected network: 3072 → 512 → 256 → 10
* Activation: ReLU
* BatchNorm + Dropout used for stability
* Custom layer: `PrunableLinear`

---

## Loss Function

Total Loss = CrossEntropy + λ × Sparsity Loss

* CrossEntropy → classification accuracy
* Sparsity Loss → encourages pruning

---

## Training

* Dataset: CIFAR-10
* Optimizer: Adam
* Epochs: 15
* Tested λ values: 1e-6, 1e-5, 5e-5

---

## Results

| Lambda | Accuracy | Sparsity |
| ------ | -------- | -------- |
| 1e-6   | ~High    | Low      |
| 1e-5   | Medium   | Medium   |
| 5e-5   | Lower    | High     |

---

## Visualizations

The following graphs are generated:

* Accuracy vs Lambda
* Sparsity vs Lambda
* Accuracy vs Sparsity (trade-off)

These clearly show how increasing λ increases sparsity but reduces accuracy.

---

## Run

```bash
pip install -r requirements.txt
python main.py
```

---

## Conclusion

The model successfully demonstrates the trade-off between **model compression (sparsity)** and **performance (accuracy)** using self-pruning.
