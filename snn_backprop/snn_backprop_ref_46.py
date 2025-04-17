import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from spikingjelly.activation_based import neuron, surrogate, functional

# For results recording and plotting
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm # For progress bar
import time # For timing

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
num_steps_train = 60   # Number of simulation steps used DURING TRAINING
batch_size = 128
learning_rate = 1e-5
num_epochs = 50 # Reduced epochs for quicker demonstration, adjust as needed
theta = 1.0        # Threshold value used for the surrogate derivative

# Data and Model Paths
data_dir = './snn_backprop/data'
model_save_path = f'./snn_backprop/models/mnist_snn_scratch_T{num_steps_train}.pth' # Save path for trained model
results_dir = './snn_backprop/results'
results_img_path = os.path.join(results_dir, f"evaluation_snn_scratch_T{num_steps_train}_sweep.png")
results_csv_path = os.path.join(results_dir, f"evaluation_snn_scratch_T{num_steps_train}_summary.csv")

os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Ensure models directory exists
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists


# Data Preparation: MNIST
transform = transforms.Compose([
    transforms.ToTensor(),  # Images normalized to [0, 1]
])
print("Loading MNIST dataset...")
try:
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Datasets and DataLoaders ready.")
except Exception as e:
    print(f"Error loading MNIST dataset: {e}")
    exit()

# Custom autograd function for output aggregation with a spatial surrogate gradient
class SpikeAggregateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike_out, z_total, theta):
        # spike_out: final binary spike output (aggregated)
        # z_total: aggregated input drive to the output layer (pre-activation)
        ctx.save_for_backward(z_total)
        ctx.theta = theta
        return spike_out # Pass through the aggregated spike count

    @staticmethod
    def backward(ctx, grad_output):
        z_total, = ctx.saved_tensors
        theta = ctx.theta
        # Surrogate derivative: 1/theta if z_total > 0, else 0.
        surrogate_grad = (z_total > 0).float() / theta
        grad_z_total = grad_output * surrogate_grad
        # No gradients for the spike_out and theta are defined.
        return None, grad_z_total, None

# Define the SNN model
class SimpleSNN(nn.Module):
    def __init__(self, theta=1.0): # Removed num_steps from init
        super(SimpleSNN, self).__init__()
        self.theta = theta

        # First layer: from input (28*28) to hidden (800)
        self.fc1 = nn.Linear(28 * 28, 800)
        # Use IFNode with surrogate gradient for backprop through hidden layer spikes
        # self.if1 = neuron.IFNode(surrogate_function=neuron.fast_sigmoid(), detach_reset=True)
        self.if1 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True) # CORRECTED

        # Second (output) layer: from hidden (800) to output (10)
        self.fc2 = nn.Linear(800, 10)
        # IFNode for output layer just generates spikes based on input drive
        # self.if2 = neuron.IFNode(surrogate_function=neuron.fast_sigmoid(), detach_reset=True)
        self.if2 = neuron.IFNode(surrogate_function=surrogate.Sigmoid(), detach_reset=True) # CORRECTED

    # Forward pass now simulates for a given number of steps 'T_sim'
    def forward(self, x, T_sim):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        # ---- Fixed Input Encoding ----
        x_fixed = (torch.rand_like(x) < x).float()

        # Reset neuron states (using functional reset externally is often preferred)
        # self.if1.reset() # Prefer functional.reset_net(model) outside
        # self.if2.reset()

        # --- Simulate BOTH layers over time ---
        out_spike_count = torch.zeros(batch_size, 10, device=x.device)
        z_out_total = torch.zeros(batch_size, 10, device=x.device)
        # Initialize membrane potential if needed by node implementation
        mem1 = None # Let IFNode handle internal state if v_init=0
        mem2 = None # Let IFNode handle internal state if v_init=0
        # Or initialize explicitly if needed:
        # mem1 = self.if1.v.new_zeros(batch_size, 800) if hasattr(self.if1, 'v') else None
        # mem2 = self.if2.v.new_zeros(batch_size, 10) if hasattr(self.if2, 'v') else None

        for _ in range(T_sim): # Simulate for T_sim steps
            # Hidden Layer
            cur1 = self.fc1(x_fixed)
            # Pass previous membrane potential for stateful update if IFNode requires it
            if mem1 is not None:
                mem1, spk1 = self.if1(cur1, mem1)
            else:
                 spk1 = self.if1(cur1) # Assumes IFNode handles state internally

            # Output Layer
            cur2 = self.fc2(spk1) # Input is instantaneous spike from hidden layer
            z_out_total += cur2 # Accumulate input drive for custom gradient
            if mem2 is not None:
                mem2, spk2 = self.if2(cur2, mem2)
            else:
                 spk2 = self.if2(cur2) # Assumes IFNode handles state internally
            out_spike_count += spk2 # Accumulate output spikes

        # Apply custom gradient function for backpropagation
        output = SpikeAggregateFunction.apply(out_spike_count, z_out_total, self.theta)
        return output

# Instantiate the model, loss function, and optimizer.
model = SimpleSNN(theta=theta).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training function
def train(model, train_loader, optimizer, criterion, num_steps_train):
    model.train()
    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    for i, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        functional.reset_net(model) # Reset states before forward pass
        outputs = model(data, T_sim=num_steps_train) # Pass training steps
        loss = criterion(outputs, target)

        loss.backward() # Calculate gradients (using custom func for output layer)
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({'Loss': loss.item()})
    return running_loss / len(train_loader)

# Testing function (MODIFIED to accept T_eval)
def test(model, test_loader, T_eval):
    model.eval()
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size_current = data.size(0)
            total += batch_size_current

            functional.reset_net(model) # Reset states before forward pass
            # Forward pass with specific evaluation timesteps (T_eval)
            # Need the aggregated spike count for prediction
            outputs = model(data, T_sim=T_eval) # Get aggregated output

            # Prediction based on highest aggregated count
            preds = outputs.argmax(dim=1)
            correct += (preds == target).sum().item()

    end_time = time.time()
    accuracy = correct / total if total > 0 else 0.0
    latency = (end_time - start_time) / total if total > 0 else 0.0
    return accuracy, latency # Return both accuracy and average latency per sample


# ==============================
# --- Main Training Phase ---
# ==============================
print(f"\n--- Training for {num_epochs} epochs ---")
for epoch in range(1, num_epochs + 1):
    start_epoch_time = time.time()
    avg_loss = train(model, train_loader, optimizer, criterion, num_steps_train)
    # Evaluate on test set after each epoch (using training num_steps for quick check)
    test_acc, _ = test(model, test_loader, T_eval=num_steps_train) # Use T_eval = num_steps_train for epoch eval
    end_epoch_time = time.time()
    print(f'Epoch: {epoch:2d} | Time: {end_epoch_time - start_epoch_time:.2f}s | Loss: {avg_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%')

# --- Save the final trained model ---
print(f"\nTraining complete. Saving final model to {model_save_path}")
torch.save(model.state_dict(), model_save_path)
print("Model saved.")


# ==============================
# --- Hyperparameter Sweep for Evaluation Timesteps (T_eval) ---
# ==============================
print(f"\n--- Starting Evaluation Sweep ---")
# Load the just-trained model (or load from file if running separately)
print(f"Loading trained model from {model_save_path}")
# Re-instantiate the model architecture
eval_model = SimpleSNN(theta=theta).to(device)
try:
    eval_model.load_state_dict(torch.load(model_save_path, map_location=device))
    print("Trained model loaded successfully for evaluation.")
except FileNotFoundError:
    print(f"ERROR: Saved model file not found at {model_save_path}. Cannot perform evaluation sweep.")
    exit()
except Exception as e:
    print(f"ERROR: Failed to load model state_dict: {e}. Cannot perform evaluation sweep.")
    exit()

# Define evaluation timesteps
eval_time_steps_list = [5, 10, 20, 30, 50, 80, 130, 210, 340, 550, 890, 1440] # Example list

# Prepare dictionary for storing results
results = {
    "T_eval": [],
    "Accuracy (%)": [],
    "Latency (s/sample)": []
}

# Perform evaluation sweep
for t_eval in eval_time_steps_list:
    print(f"\nEvaluating for T_eval = {t_eval}...")
    # Call the modified test function with specific T_eval
    acc, lat = test(eval_model, test_loader, T_eval=t_eval)
    results["T_eval"].append(t_eval)
    results["Accuracy (%)"].append(acc * 100.0) # Store as percentage
    results["Latency (s/sample)"].append(lat)

# Convert results to DataFrame and save/print
df_results = pd.DataFrame(results)
print("\n--- Evaluation Sweep Results ---")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(df_results)
df_results.to_csv(results_csv_path, index=False)
print(f"\nEvaluation results saved to {results_csv_path}")


# ==============================
# --- Plotting Results ---
# ==============================
print("\nGenerating evaluation plots...")
try:
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot 1: Accuracy vs. Evaluation timesteps
    sns.lineplot(data=df_results, x="T_eval", y="Accuracy (%)", marker='o', ax=axs[0])
    axs[0].set_xlabel("Evaluation Timesteps (T_eval)")
    axs[0].set_ylabel("Accuracy (%)")
    axs[0].set_title("Accuracy vs. Evaluation Timesteps")
    axs[0].grid(True, which="both", linestyle="--", alpha=0.6)

    # Subplot 2: Latency vs. Evaluation timesteps
    sns.lineplot(data=df_results, x="T_eval", y="Latency (s/sample)", marker='o', ax=axs[1])
    axs[1].set_xlabel("Evaluation Timesteps (T_eval)")
    axs[1].set_ylabel("Avg. Latency per Sample (s)")
    axs[1].set_title("Latency vs. Evaluation Timesteps")
    axs[1].grid(True, which="both", linestyle="--", alpha=0.6)

    plt.suptitle(f"SNN Scratch Training (T_train={num_steps_train}) Evaluation", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(results_img_path)
    print(f"Plots saved to {results_img_path}")
    # plt.show()
    plt.close()

except Exception as e_plot:
    print(f"Error generating plots: {e_plot}")
    print("Ensure pandas, matplotlib, and seaborn are installed.")

print("\nScript finished.")