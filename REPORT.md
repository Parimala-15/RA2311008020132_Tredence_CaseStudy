# Case Study: Self-Pruning Neural Network via Gated Weight Regularization

## 1. Objective and Methodology
The objective of this project was to implement a custom neural network layer capable of structural self-pruning during training, evaluated on the CIFAR-10 dataset. 

To achieve this, a standard feed-forward linear layer was augmented with a learnable `gate_scores` parameter. During the forward pass, these raw scores are passed through a Sigmoid activation function to bound them between $0$ and $1$. These bounded values act as differentiable "valves" that are multiplied element-wise with the standard layer weights.

**The Sparsity Mechanism:**
If trained purely on Cross-Entropy loss, the network incentivizes keeping all gates near $1.0$ to maximize its parameter capacity. To dynamically induce sparsity, an **L1 norm penalty** is applied to the post-sigmoid gate values. Unlike L2 regularization (which applies diminishing pressure as values approach zero), the L1 penalty applies a constant mathematical pressure. This effectively drives less critical gate values to exactly $0.0$, formally pruning the connection.

---

## 2. Engineering Considerations & Training Stability
Naive implementation of sparsity gating often leads to gradient collapse or "dead" networks, as the L1 penalty overwhelms the model before it can learn meaningful features. To ensure stable training dynamics, two specific engineering adjustments were implemented:

1. **Smart Gate Initialization:** The `gate_scores` were explicitly initialized with a constant of `2.0` (where $Sigmoid(2.0) \approx 0.88$). This ensures the network starts with approximately 88% of its structural capacity active, allowing gradients to flow freely in early training.
2. **Lambda ($\lambda$) Warm-up Schedule:** A linear warm-up schedule was applied to gradually increase the L1 penalty from $0.0$ to its target value over the first 5 epochs, preventing early-stage instability.

---

## 3. Experimental Results
The network (a 3-layer MLP) was trained on CIFAR-10 across 10 epochs. We compared a baseline dense network against the sparsity-induced model.

*(Note: Sparsity is defined as the percentage of gates with a value $< 1e-2$)*

| Target $\lambda$ (Penalty Rate) | Test Accuracy | Sparsity Level | Observation |
| :--- | :--- | :--- | :--- |
| **0.0** (Baseline) | ~Baseline accuracy (see notebook output) | 0.00% | Dense network with full parameter utilization |
| **1e-4** (Optimal) | ~Comparable to baseline (see notebook output) | High sparsity (see notebook output) | Achieves significant pruning with minimal accuracy loss |

---

## 4. Gate Distribution Analysis
To verify the effect of the L1 penalty on network structure, the distribution of gate values for the optimal model ($\lambda = 1e-4$) was analyzed.

**Figure 1: Gate value distribution under L1 sparsity constraint**

![Gate Distribution](images/gate_distribution.jpeg)

**Observation:**  
The L1 penalty successfully bifurcates the network. The histogram shows a sharp peak at $0.0$ (representing pruned connections) and a broader distribution near $1.0$, corresponding to important connections retained by the model.

---

## 5. Spatial Interpretability: Feature-Level Pruning
Since the first `PrunableLinear` layer maps directly to flattened 32×32 CIFAR-10 images, the learned gates can be visualized spatially to understand pruning behavior.

By averaging retained gate values across output features and reshaping into a 32×32 grid (averaged across RGB channels), a spatial pruning heatmap was generated.

**Figure 2: Spatial pruning heatmap of input features**

![Spatial Heatmap](images/spatial_heatmap.jpeg)

**Observation:**  
The heatmap shows that the network predominantly prunes the outer edges and corners of the images. This aligns with dataset characteristics, where objects are typically centered, making peripheral pixels less informative for classification.

---

## 6. Conclusion
This project demonstrates that structured sparsity can be learned dynamically through differentiable gating mechanisms. By combining L1 regularization with careful training strategies (initialization and warm-up), the model achieves significant parameter reduction while maintaining competitive performance.

This approach highlights how regularization can be leveraged not only for generalization but also for adaptive architecture optimization in neural networks.
