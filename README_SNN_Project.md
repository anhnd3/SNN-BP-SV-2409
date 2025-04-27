
# Training Deep Spiking Neural Networks: ANN-to-SNN Conversion and Direct Backpropagation

## 🧠 Introduction
This project explores two primary methods for training deep Spiking Neural Networks (SNNs):
- **ANN-to-SNN Conversion**: Leveraging pretrained ANN models and converting them to event-driven SNNs.
- **Direct Backpropagation (Spatial Approach)**: Training SNNs from scratch using surrogate gradients while optimizing memory efficiency.

We performed systematic experiments on CIFAR-10 and MNIST datasets using SpikingJelly framework, evaluating the trade-offs between accuracy, latency, and network structure.

## ⚙️ Installation
```bash
pip install torch torchvision
pip install spikingjelly
pip install matplotlib pandas tqdm
```

Recommended environment: Python 3.8+, PyTorch 1.10+, GPU preferred for SNN training.

## 🚀 How to Run Experiments

### ANN-to-SNN Conversion (Rate Coding)
1. Train VGG-16 ANN on CIFAR-10 (or load pre-trained weights).
2. Convert ANN to SNN using SpikingJelly's `ann2snn` module.
3. Evaluate on CIFAR-10 using fixed T_eval timesteps.

### Direct SNN Training (Spatial Approach)
1. Train 3-layer CNN SNN directly on MNIST dataset.
2. Use Spatial Surrogate Gradient at output (Spike Count loss).
3. Evaluate accuracy vs latency at different evaluation timesteps.

## 📈 Results Summary

| Experiment | Accuracy (%) | Timestep (T_eval) | Latency (s/sample) |
|:---|:---|:---|:---|
| ANN-to-SNN (VGG-16, CIFAR-10) | ~88.0% | 30–40 | ~0.08 |
| Spatial Backprop (3-layer CNN, MNIST) | ~96.6% | 40–50 | < 0.0002 |

## 🔍 Insights & Lessons Learned
- ANN-to-SNN conversion suffers ~3–4% accuracy drop mainly due to quantization and unevenness error.
- Direct Spatial training achieves higher accuracy with moderate training cost.
- Threshold balancing and fine-tuning significantly affect final SNN performance.
- Hybrid coding strategies could further reduce latency without sacrificing accuracy.

## 📅 Future Work
- Implement hybrid Rate + Temporal coding in ANN-to-SNN pipeline.
- Explore soft reset neurons for direct SNN training.
- Expand architecture depth for MNIST and CIFAR datasets.
- Energy efficiency benchmarking on neuromorphic hardware.

## 📜 References
- [44] J. H. Lee, T. Delbruck, M. Pfeiffer, “Training Deep SNNs using Backpropagation,” Frontiers in Neuroscience, 2016.
- [45] J. C. Thiele, O. Bichler, A. Dupret, “SpikeGrad: An ANN-equivalent model for backprop with spikes,” arXiv:1906.00851, 2019.
- [46] J. Wu et al., “Deep SNN with Spike Count based Learning Rule,” IJCNN 2019.
- Comprehensive Survey Paper on Deep SNN Training (2022).
