import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, surrogate, functional
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------
# 0) CONFIG & HYPERPARAMS
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

T_train    = 60       # number of timesteps during training
batch_size = 64
lr         = 1e-5
epochs     = 150
theta      = 1.0      # fixed threshold
weight_decay = 1e-4   # L2 regularization as in [46]

data_dir   = "./data"


os.makedirs(data_dir, exist_ok=True)

# ---------------------
# 1) DATA
# ---------------------
transform = transforms.Compose([transforms.ToTensor()])
train_ds  = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
test_ds   = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

# --------------------------------------------------
# 2) SPIKE AGGREGATION WITH UNIFORM SURROGATE GRADIENT
# --------------------------------------------------
class SpikeAggregateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_count, z_total, theta):
        ctx.save_for_backward(z_total)
        ctx.theta = theta
        return spike_count

    @staticmethod
    def backward(ctx, grad_out):
        z_total, = ctx.saved_tensors
        theta = ctx.theta
        # surrogate: d(count)/d(z) = 1/theta if z>0 else 0
        sur = (z_total > 0).float() / theta
        return None, grad_out * sur, None

# ---------------------
# 3) MODEL DEFINITION
# ---------------------
class WuSpatialCNN(nn.Module):
    def __init__(self, theta=1.0):
        super().__init__()
        self.theta = theta
        # conv block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True) # Add bias
        self.bn1 = nn.BatchNorm2d(32) # Add BatchNorm
        self.if1 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        # conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True) # Add bias
        self.bn2 = nn.BatchNorm2d(64) # Add BatchNorm
        self.if2 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

        # fully-connected
        self.fc = nn.Linear(64 * 7 * 7, 10, bias=True) # Add bias
        self.if3 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True)

    def forward(self, x, T_sim):
        B = x.size(0)
        functional.reset_net(self)

        out_count = torch.zeros(B, 10, device=x.device)
        z_total = torch.zeros(B, 10, device=x.device)

        for _ in range(T_sim):
            # conv1
            c1 = self.conv1(x)
            c1_bn = self.bn1(c1) # Apply BatchNorm
            s1 = self.if1(c1_bn)
            p1 = nn.functional.max_pool2d(s1, 2)

            # conv2
            c2 = self.conv2(p1)
            c2_bn = self.bn2(c2) # Apply BatchNorm
            s2 = self.if2(c2_bn)
            p2 = nn.functional.max_pool2d(s2, 2)

            # fc
            feat = p2.view(B, -1)
            cur3 = self.fc(feat)
            # Consider adding BN before the last IFNode as well if it helps
            # cur3_bn = self.bn3(cur3) # Optional: Add BN for FC output

            z_total += cur3 # Accumulate pre-activation (matches paper's z_i^n definition somewhat)
            # If using BN on cur3, accumulate cur3_bn instead: z_total += cur3_bn

            s3 = self.if3(cur3) # Spikes from final layer (use cur3 or cur3_bn)
            out_count += s3

        return SpikeAggregateFunction.apply(out_count, z_total, self.theta) # Ensure z_total matches the accumulated value fed to IFNode

# ---------------------
# 4) TRAIN & EVAL
# ---------------------
model     = WuSpatialCNN(theta).to(device)

# Apply Kaiming Normal initialization
for m in model.modules():
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
             nn.init.constant_(m.bias, 0) # Initialize bias to 0


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def train_epoch():
    model.train()
    running_loss = 0.0
    for x, y in tqdm(train_loader, desc="Train"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x, T_train)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

@torch.no_grad()
def eval_acc(T):
    model.eval()
    correct = 0
    for x, y in tqdm(test_loader, desc=f"Eval @T={T:2d}"):
        x, y = x.to(device), y.to(device)
        out = model(x, T)
        preds = out.argmax(dim=1)
        correct += (preds == y).sum().item()
    return correct / len(test_loader.dataset)

# ---------------------
# 5) MAIN LOOP
# ---------------------
history = {"epoch": [], "loss": [], "acc": []}
for ep in range(1, epochs+1):
    t0 = time.time()
    loss = train_epoch()
    acc  = eval_acc(T_train)
    dt  = time.time() - t0
    print(f"Epoch {ep:2d}/{epochs} — loss: {loss:.4f} — acc: {acc*100:5.2f}% — {dt:.1f}s")
    history["epoch"].append(ep)
    history["loss"].append(loss)
    history["acc"].append(acc)

# ---------------------
# 6) HYPERPARAM SWEEP
# ---------------------
eval_T = [5,10,20,30,40,50,60,80,100]
results = []
for T in eval_T:
    acc = eval_acc(T)
    results.append({"T_eval":T, "Accuracy(%)":acc*100})
    print(f"T={T:3d} → {acc*100:5.2f}%")

df = pd.DataFrame(results)
print(df)

# ---------------------
# 7) VISUALIZE
# ---------------------
plt.figure(figsize=(6,4))
sns.lineplot(data=df, x="T_eval", y="Accuracy(%)", marker="o")
plt.title("Wu et al. [46] Spatial CNN on MNIST")
plt.xlabel("T_eval")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.tight_layout()
plt.show()

# import os
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader
# from spikingjelly.activation_based import neuron, surrogate, functional
# from tqdm import tqdm
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # 1) Configuration & Hyperparameters
# device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Device:", device)

# T_train     = 60        # simulation steps during training
# batch_size  = 64
# learning_rate = 1e-5
# epochs      = 150
# theta       = 1.0       # threshold for surrogate gradient

# # Paths
# data_dir    = "./data"
# results_dir = "./results"
# os.makedirs(results_dir, exist_ok=True)

# # 2) Data preparation (MNIST)
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])
# train_ds = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
# test_ds  = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
# test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)

# # 3) Custom surrogate aggregated over final spike count
# class SpikeAggregateFunction(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, out_cnt, z_tot, theta):
#         ctx.save_for_backward(z_tot)
#         ctx.theta = theta
#         return out_cnt

#     @staticmethod
#     def backward(ctx, grad_out):
#         z_tot, = ctx.saved_tensors
#         theta = ctx.theta
#         sur = (z_tot > 0).float() / theta
#         return None, grad_out * sur, None

# # 4) IFNode that records pre-activation
# class IFWithZ(neuron.IFNode):
#     def forward(self, x):
#         self.last_z = x
#         return super().forward(x)

# # 5) Spatial CNN-SNN
# sur_fn = surrogate.Sigmoid()  # surrogate gradient

# class SpatialCNN(nn.Module):
#     def __init__(self, theta=1.0):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, padding=1, bias=False)
#         self.bn1   = nn.BatchNorm2d(32)
#         self.if1   = IFWithZ(surrogate_function=sur_fn, detach_reset=True)
#         self.pool1 = nn.MaxPool2d(2)

#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
#         self.bn2   = nn.BatchNorm2d(64)
#         self.if2   = IFWithZ(surrogate_function=sur_fn, detach_reset=True)
#         self.pool2 = nn.MaxPool2d(2)

#         self.fc    = nn.Linear(64 * 7 * 7, 10, bias=False)
#         self.if3   = IFWithZ(surrogate_function=sur_fn, detach_reset=True)

#     def _forward_once(self, x):
#         z1 = self.conv1(x);  self.bn1(z1);  s1 = self.if1(z1);  p1 = self.pool1(s1)
#         z2 = self.conv2(p1); self.bn2(z2);  s2 = self.if2(z2);  p2 = self.pool2(s2)
#         f  = p2.view(x.size(0), -1)
#         z3 = self.fc(f);                      
#         s3 = self.if3(z3)
#         return s3, z3

#     def forward(self, x, T_sim):
#         B = x.size(0)
#         functional.reset_net(self)
#         out_cnt = torch.zeros(B, 10, device=x.device)
#         z_tot   = torch.zeros(B, 10, device=x.device)
#         for _ in range(T_sim):
#             # Poisson rate encoding
#             x_spk = (torch.rand_like(x) < x).float()
#             s3, z3 = self._forward_once(x_spk)
#             out_cnt += s3
#             z_tot   += z3
#         return SpikeAggregateFunction.apply(out_cnt, z_tot, theta)

# # 6) Instantiate and initialize
# model = SpatialCNN(theta).to(device)
# # He initialization
# for m in model.modules():
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

# # 7) Threshold calibration
# def calibrate_thresholds(model, loader, T=20):
#     # Only track layers that actually have weights and that we initialized
#     max_vals = {}
#     for name, m in model.named_modules():
#         if isinstance(m, (nn.Conv2d, nn.Linear)):
#             max_vals[name] = 1e-12

#     model.eval()
#     with torch.no_grad():
#         for x, _ in loader:
#             # Poisson‐encode once to get x_spk
#             x_spk = (torch.rand_like(x) < x).float().to(device)
#             functional.reset_net(model)
#             for _ in range(T):
#                 # call the internal forward to populate last_z on IFWithZ nodes
#                 _ = model._forward_once(x_spk)
#             # now update only the tracked layers
#             for name, m in model.named_modules():
#                 if name in max_vals and hasattr(m, "last_z"):
#                     max_vals[name] = max(max_vals[name], m.last_z.abs().max().item())

#     # finally normalize weights of the tracked layers
#     for name, m in model.named_modules():
#         if name in max_vals and isinstance(m, (nn.Conv2d, nn.Linear)):
#             m.weight.data /= (max_vals[name] / theta)

# calibrate_thresholds(model, train_loader, T=20)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # 8) Training loop
# for ep in range(1, epochs+1):
#     model.train()
#     sum_loss = 0.0
#     for x, y in tqdm(train_loader, desc=f"Train Ep{ep}/{epochs}"):
#         x, y = x.to(device), y.to(device)
#         optimizer.zero_grad()
#         out = model(x, T_train)
#         loss = criterion(out, y)
#         loss.backward()
#         optimizer.step()
#         sum_loss += loss.item()
#     # quick eval on T_train
#     correct = 0
#     with torch.no_grad():
#         for x,y in test_loader:
#             x,y = x.to(device), y.to(device)
#             pred = model(x, T_train).argmax(1)
#             correct += (pred==y).sum().item()
#     acc = correct / len(test_ds)
#     print(f"Epoch {ep}/{epochs}  loss={sum_loss/len(train_loader):.3f}  acc={acc*100:.2f}%")

# # 9) Evaluation sweep & plotting
# eval_T = [5, 10, 20, 30, 40, 50, 60, 80, 100, 130, 210]
# results = []
# for T in eval_T:
#     start = time.time()
#     correct = 0
#     with torch.no_grad():
#         for x,y in test_loader:
#             x,y = x.to(device), y.to(device)
#             pred = model(x, T).argmax(1)
#             correct += (pred==y).sum().item()
#     acc = correct / len(test_ds)
#     lat = (time.time() - start) / len(test_ds)
#     results.append({'T_eval': T, 'Accuracy (%)': acc*100, 'Latency (s/sample)': lat})
#     print(f"T={T:3d} -> acc={acc*100:.2f}%  lat={lat*1e3:.3f}ms")

# df = pd.DataFrame(results)
# df.to_csv(os.path.join(results_dir, "mnist_spatial_sweep.csv"), index=False)
# sns.lineplot(data=df, x='T_eval', y='Accuracy (%)', marker='o')
# plt.title("MNIST Accuracy vs T_eval")
# plt.grid(True)
# plt.savefig(os.path.join(results_dir, "accuracy_vs_Teval.png"))
