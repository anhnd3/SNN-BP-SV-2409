# --- Imports ---
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights
from torch.cuda.amp import GradScaler, autocast # For AMP
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import copy # For deep copying model

# --- SpikingJelly Imports (for neuron and functional) ---
# Note: We are primarily using SpikingJelly for LIFNode and functional.reset_net
# The model structure itself uses standard PyTorch layers wrapped by LIFNode.
from spikingjelly.clock_driven import functional, neuron, surrogate

# --- Tonic Imports (for data loading) ---
# Requires: pip install tonic
try:
    import tonic
    import tonic.transforms as T_transforms # Use alias
    # Verify exact class name in Tonic documentation if 'CIFAR10DVS' causes issues
    from tonic.datasets import CIFAR10DVS
    # from tonic.collation import PadTensors # May need if using variable length event streams
except ImportError:
    print("Error: Tonic library not found. Please install it: pip install tonic")
    exit()

# Standard PyTorch DataLoader
from torch.utils.data import DataLoader
# For PIL conversion if needed in transforms
import torchvision.transforms.functional as F_tv


# -------------------------------
# Configuration
# -------------------------------
# --- Experiment Params ---
coding_type = "TTFS"
dataset_name = "CIFAR10DVS"

# --- Fine-Tuning Params ---
finetune_snn = True # Enable fine-tuning
finetune_epochs = 1 # Number of fine-tuning epochs (Adjust as needed)
learning_rate = 1e-4 # Learning rate (Adjust as needed)
optimizer_choice = 'AdamW'

# --- SNN Simulation Params ---
# CRUCIAL: Select T based on dataset properties and desired temporal resolution
T = 32 # Total timesteps (Adjust as needed)
# dt = 1e-3 # Simulation time step (often implicit in T)

# --- Data/Model Params ---
num_classes = 10 # CIFAR10-DVS has 10 classes
# Adjust batch size based on GPU memory
batch_size = 8
data_dir = './ann_to_snn/data' # Root data directory to download/load datasets
ann_model_name = "vgg16"
# << Path for the TTFS fine-tuned SNN model >>
model_save_path = f'./ann_to_snn/data/models/{ann_model_name}_{dataset_name}_T{T}_{coding_type}_finetuned.pth'
best_model_save_path = model_save_path.replace('.pth', '_best.pth')
os.makedirs(os.path.dirname(model_save_path), exist_ok=True) # Ensure directory exists

# -------------------------------
# Device Setup
# -------------------------------

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using NVIDIA CUDA device.")
else:
    device = torch.device("cpu")
    print("Using CPU device.")
    
print("-" * 50)
print("CUDA Diagnostics:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    print(f"torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}") # Check name of device 0
print(f"Device variable used in script: {device}")
print("-" * 50)    

print("PyTorch Version:", torch.__version__)
# Adjust num_workers based on your system's RAM and CPU cores
num_workers = 0 # Start moderate, set to 0 if memory/forking issues occur
pin_memory = False if device.type == 'cuda' else False
print(f"Using Device: {device}, Num Workers: {num_workers}, Pin Memory: {pin_memory}")

# --- AMP (Automatic Mixed Precision) Setup ---
use_amp = torch.cuda.is_available() # Enable AMP only if CUDA is available
# Initialize GradScaler OUTSIDE the training function
scaler = torch.amp.GradScaler(device='cuda', enabled=use_amp)
print(f"Using Automatic Mixed Precision (AMP): {use_amp}")

###############################################_
# Spiking Wrapper Module (from User Script)
# Wraps a layer (usually Identity after Conv/Linear)
# with a LIF neuron.
###############################################
class SpikingWrapper(nn.Module):
    def __init__(self, layer, v_threshold=1.0, sg_func=surrogate.Sigmoid()): # Using Sigmoid surrogate like user script
        super().__init__()
        self.layer = layer # Typically nn.Identity here
        self.neuron = neuron.LIFNode(v_threshold=v_threshold,
                                      surrogate_function=sg_func,
                                      detach_reset=True) # detach_reset often helps stability

    def forward(self, x):
        # Compute linear transformation (or identity) then spike
        # Important: Reset neuron state externally using functional.reset_net before simulation loop
        out = self.layer(x)
        out = self.neuron(out)
        return out

###############################################
# ANN to SNN Conversion (Direct Weight Mapping via ReLU Replacement - from User Script)
###############################################
def convert_ann_to_snn(ann_model):
    print("Converting ANN to SNN by replacing ReLUs with SpikingWrapper(LIFNode)...")
    snn_model = copy.deepcopy(ann_model) # Start with ANN structure and weights

    def replace_relu_recursive(module):
        for name, child_module in module.named_children():
            if isinstance(child_module, nn.ReLU):
                # Replace ReLU with SpikingWrapper(Identity)
                setattr(module, name, SpikingWrapper(nn.Identity()))
            elif isinstance(child_module, nn.Dropout):
                 # Option: Remove dropout or keep it. Keeping it for now.
                 print(f"Keeping Dropout layer: {name}")
                 pass # Keep dropout layer
            else:
                # Recursively apply to child modules
                replace_relu_recursive(child_module)

    replace_relu_recursive(snn_model.features)
    replace_relu_recursive(snn_model.classifier)

    # Ensure final classifier layer is Linear (no activation/spike needed for logits)
    if isinstance(snn_model.classifier[-1], SpikingWrapper):
         print("Replacing final SpikingWrapper in classifier with original Linear layer.")
         # Find the original Linear layer it replaced
         original_linear = None
         for layer in ann_model.classifier: # Check original ANN
             if isinstance(layer, nn.Linear):
                  original_linear = layer # Assumes last linear is the one
         if original_linear:
              snn_model.classifier[-1] = original_linear
         else: # Fallback
              print("Error: Could not find original final Linear layer. Manual check needed.")
              # As a fallback, try unwrapping (might be incorrect if structure changed)
              if hasattr(snn_model.classifier[-1],'layer') and isinstance(snn_model.classifier[-1].layer, nn.Linear):
                   snn_model.classifier[-1] = snn_model.classifier[-1].layer

    elif not isinstance(snn_model.classifier[-1], nn.Linear):
        print(f"Warning: Final classifier layer is not Linear: {type(snn_model.classifier[-1])}. Check model.")


    print("ReLU replacement complete.")
    return snn_model

###############################################
# Utility: Get First Spike Times (from User Script)
###############################################
def get_first_spike_times(out_seq, T):
    # out_seq shape: [T, B, output_dim]
    # Check for valid input dimensions
    if out_seq is None or out_seq.dim() < 3:
        print(f"Warning: Invalid input to get_first_spike_times (shape={out_seq.shape if out_seq is not None else 'None'}). Returning dummy times.")
        # Attempt to create a dummy tensor based on expected B, N if possible, otherwise fail
        try:
             # This part is hard without Batch dim info, maybe return None or raise Error
             # Returning a single dummy value for now
             return torch.tensor([[float(T + 1)]], device=out_seq.device if out_seq is not None else 'cpu')
        except:
             return None # Indicate failure

    t_dim, b_dim, out_dim = out_seq.shape
    first_spike_times = torch.full((b_dim, out_dim), float(T + 1), device=out_seq.device)

    for t in range(t_dim): # Iterate over actual time dimension length
        # Assuming out_seq contains potentials/activations, use threshold
        spikes_t = (out_seq[t] > 0).float() # [B, output_dim]
        # Identify neurons that spiked now *for the first time*
        mask = (first_spike_times == float(T + 1)) & (spikes_t == 1)
        first_spike_times[mask] = t # Record timestep 't' (using index, assumes dt=1 step)

    return first_spike_times # [B, output_dim]

###############################################
# Visualization Helpers (from User Script)
###############################################
def plot_spike_raster(out_seq, gt_label, pred_label, sample_idx=0, title_prefix=""):
    if out_seq is None or out_seq.dim() < 3 or out_seq.shape[1] <= sample_idx:
        print(f"Cannot plot raster for sample {sample_idx}: Invalid data.")
        return
    spikes = out_seq[:, sample_idx].detach().cpu()
    num_neurons = spikes.size(1)
    if num_neurons == 0: return

    plt.figure(figsize=(10, max(4, num_neurons * 0.3))) # Adjust height dynamically
    neuron_indices = torch.arange(num_neurons)
    colors = plt.cm.viridis(neuron_indices / max(1, num_neurons - 1)) # Color code neurons

    for neuron_idx in neuron_indices:
        spike_times = torch.nonzero(spikes[:, neuron_idx] > 0).squeeze(-1) # Use > 0
        if spike_times.numel() > 0:
             plt.vlines(spike_times, neuron_idx + 0.5, neuron_idx + 1.5, color=colors[neuron_idx], linewidth=1.5)

    plt.title(f"{title_prefix}Spike Raster | Sample: {sample_idx}, GT: {gt_label}, Pred: {pred_label}")
    plt.xlabel("Timestep (T)")
    plt.ylabel("Output Neuron Index")
    if num_neurons > 0: plt.yticks(range(num_neurons))
    plt.ylim(0.0, num_neurons + 0.5) # Adjust y-limits
    plt.gca().invert_yaxis() # Show neuron 0 at top
    plt.grid(True, axis='x', linestyle=':', alpha=0.7)
    plt.tight_layout()
    try:
        plt.savefig(f"{title_prefix}spike_raster_sample{sample_idx}.png")
    except Exception as e:
        print(f"Failed to save spike raster plot: {e}")
    plt.close()

def plot_spike_count_hist(spike_counts, title="Spike Count Histogram"):
    if spike_counts is None:
        print("Cannot plot histogram: Invalid data.")
        return
    num_neurons = len(spike_counts)
    if num_neurons == 0: return

    plt.figure(figsize=(max(8, num_neurons * 0.5), 4)) # Adjust width dynamically
    plt.bar(range(num_neurons), spike_counts)
    plt.xlabel("Output Neuron")
    plt.ylabel("Total Spike Count")
    plt.title(title)
    if num_neurons > 0: plt.xticks(range(num_neurons))
    plt.grid(True, axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()
    try:
        # Sanitize title for filename
        safe_title = "".join(c if c.isalnum() else "_" for c in title)
        plt.savefig(f"{safe_title}.png")
    except Exception as e:
        print(f"Failed to save histogram plot: {e}")
    plt.close()

###############################################
# DATA LOADING: CIFAR10-DVS using Tonic
# NOTE: VERIFY Class Name and Arguments from Tonic Docs!
# Ensure Tonic is installed: pip install tonic
###############################################
print(f"Setting up {dataset_name} DataLoader using Tonic (T={T})...")

# --- Tonic Transforms ---
# Attempt to get sensor size dynamically, fallback to default
try:
    sensor_size = tonic.datasets.CIFAR10DVS.sensor_size # e.g., (128, 128)
    if isinstance(sensor_size, tuple) and len(sensor_size) == 2:
        sensor_size = (sensor_size[0], sensor_size[1], 2) # Append channels (H, W, C) format for ToFrame
    elif not isinstance(sensor_size, tuple) or len(sensor_size) != 3:
        print("Warning: Could not determine sensor size automatically from Tonic. Assuming 128x128x2.")
        sensor_size = (128, 128, 2)
except AttributeError:
    print("Warning: tonic.datasets.CIFAR10DVS.sensor_size not found. Assuming 128x128x2.")
    sensor_size = (128, 128, 2)

print(f"Using sensor size: {sensor_size} for Tonic ToFrame")

# Define transforms: Convert events to frames, then resize frames.
# Using ToFrame outputs tensors (usually). torchvision Resize works on tensors [..., H, W].
frame_transform = T_transforms.Compose([
    # T_transforms.Denoise(filter_time=10000), # Optional
    T_transforms.ToFrame(sensor_size=sensor_size, n_time_bins=T)
    # REMOVED transforms.Resize from here
])

# --- Create Full Tonic Dataset ---
try:
    print(f"Loading Full {dataset_name} from: {data_dir}")
    full_dataset_tonic = CIFAR10DVS(
        save_to=data_dir,
        transform=frame_transform
    )
    print(f"Tonic full dataset object created. Total samples: {len(full_dataset_tonic)}")
    if len(full_dataset_tonic) == 0:
         raise ValueError("Loaded dataset is empty!")
    # Common split: 9000 train, 1000 test (total 10000) - VERIFY THIS!
    num_total_samples = len(full_dataset_tonic)
    # Adjust these numbers if the total count or standard split differs
    num_train_samples = 9000
    num_test_samples = 1000

    if num_total_samples != (num_train_samples + num_test_samples):
         print(f"Warning: Expected total {num_train_samples + num_test_samples} samples for split, but found {num_total_samples}.")
         # Fallback: Use a percentage split (e.g., 90/10)
         num_train_samples = int(0.9 * num_total_samples)
         num_test_samples = num_total_samples - num_train_samples
         print(f"Using fallback split: Train={num_train_samples}, Test={num_test_samples}")


    # --- Split into Train and Test using torch.utils.data.Subset ---
    # Assumes the first num_train_samples are the training set
    print(f"Splitting into Train: {num_train_samples}, Test: {num_test_samples} using Subset...")
    train_indices = list(range(num_train_samples))
    test_indices = list(range(num_train_samples, num_train_samples + num_test_samples))

    train_dataset_tonic = torch.utils.data.Subset(full_dataset_tonic, train_indices)
    test_dataset_tonic = torch.utils.data.Subset(full_dataset_tonic, test_indices)

    print(f"Train subset size: {len(train_dataset_tonic)}")
    print(f"Test subset size: {len(test_dataset_tonic)}")


    # --- Alternative: Random Split (if order is not fixed or you want random subsets) ---
    # generator = torch.Generator().manual_seed(42) # for reproducibility
    # train_dataset_tonic, test_dataset_tonic = torch.utils.data.random_split(
    #     full_dataset_tonic, [num_train_samples, num_test_samples], generator=generator
    # )
    # print("Split dataset using random_split.")


except NameError:
    print(f"Error: Class 'CIFAR10DVS' not found or not imported correctly from tonic.datasets.")
    exit()
except Exception as e:
     print(f"Error loading/splitting {dataset_name} via Tonic: {e}")
     exit()

def worker_init_fn(worker_id):
    # Ensure every DataLoader worker uses device cuda:0.
    torch.cuda.set_device(0)

# --- Create Standard PyTorch DataLoaders ---
train_loader = DataLoader(
    train_dataset_tonic, batch_size=batch_size, shuffle=True, # Shuffle training set
    num_workers=num_workers, pin_memory=pin_memory, drop_last=True,
    worker_init_fn=worker_init_fn
)
test_loader = DataLoader(
    test_dataset_tonic, batch_size=batch_size, shuffle=False, # No shuffle for test set
    num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
    worker_init_fn=worker_init_fn
)
print(f"Tonic DataLoaders ready (Batch Size: {batch_size}).")

# --- Create Standard PyTorch DataLoaders ---
# Default collate should work if dataset yields fixed-size tensors [T, C, H, W]
train_loader = DataLoader(
    train_dataset_tonic, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=pin_memory, drop_last=True
)
test_loader = DataLoader(
    test_dataset_tonic, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=pin_memory, drop_last=False
)
print(f"Tonic DataLoaders ready (Batch Size: {batch_size}).")

# Debug: Check data shape after loading one batch
time_dim_index = -1 # Default to unknown
try:
    print("Checking sample batch shape from Tonic loader...")
    data_iter = iter(train_loader)
    sample_batch, sample_labels = next(data_iter) # Loader yields tuple(data, label)
    print("Sample batch shape:", sample_batch.shape)
    print("Sample labels shape:", sample_labels.shape)

    # Determine time dimension index based on shape and T
    if len(sample_batch.shape) >= 4: # Need at least T, C, H, W or B, T, C, H, W
        if sample_batch.shape[0] == T:
             time_dim_index = 0
             print("Detected time dimension first: [T, B, C, H, W]")
        elif sample_batch.shape[1] == T:
             time_dim_index = 1
             print("Detected time dimension second: [B, T, C, H, W]")
        else:
             print(f"Warning: Could not detect time dimension T={T} in shape {sample_batch.shape}")
    else:
        print(f"Warning: Unexpected batch shape, expected >= 4 dims: {sample_batch.shape}")

except StopIteration:
    print("Could not load a sample batch: DataLoader is empty (check dataset).")
    exit()
except Exception as e:
    print(f"Could not load a sample batch via Tonic: {e}")
    exit()


###############################################
# LOAD PRETRAINED ANN (VGG16) and Convert to SNN
###############################################
print("Loading ANN and converting to SNN...")
ann_model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
# Modify the final classifier for CIFAR-10 (10 classes).
# Ensure this matches the SNN output dimensions
ann_model.classifier[-1] = nn.Linear(ann_model.classifier[-1].in_features, num_classes)
# Use the conversion function based on replacing ReLUs
snn_model = convert_ann_to_snn(ann_model).to(device)
print(f"SNN model created using conversion method and moved to {device}.")


###############################################
# TTFS Training Function (Resizing moved inside)
###############################################
def train_ttfs(model, train_loader, test_loader, T, device, epochs, lr, model_save_path, best_model_save_path, time_dim_idx, loss_margin=1.0, loss_no_spike_penalty=1.0):
    # Assuming global scaler and use_amp are defined
    global scaler, use_amp

    print("\nStarting TTFS SNN Fine-tuning (with isolated .to(device) calls)...")
    print(f" Params: Epochs={epochs}, LR={lr}, T={T}, BatchSize={train_loader.batch_size}")

    # model.to(device) # Model should be moved to device BEFORE calling this function
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() # Uses logits = -first_spike_times
    resizer = transforms.Resize((224, 224), antialias=True) # Define resizer once

    best_val_acc = -1.0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train() # Ensure model is in train mode at start of epoch
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        start_epoch_time = time.time()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")
        for i, batch_data in pbar:
            # --- Data Loading and Initial Prep ---
            if isinstance(batch_data, (list, tuple)):
                inputs_cpu, labels_cpu = batch_data[0], batch_data[1] # Keep on CPU first
            else:
                print(f"Warning: Unexpected batch data format (type: {type(batch_data)}). Skipping batch {i}.")
                continue

            B = labels_cpu.size(0) if isinstance(labels_cpu, torch.Tensor) else 0
            if B == 0: continue
            total_train += B # Increment total samples count early

            # Convert if NumPy (should happen before moving)
            if not isinstance(inputs_cpu, torch.Tensor):
                 if isinstance(inputs_cpu, np.ndarray):
                     inputs_cpu = torch.from_numpy(inputs_cpu)
                 else:
                     print(f"Warning: Input batch type {type(inputs_cpu)} not tensor or numpy. Skipping batch {i}.")
                     total_train -= B; continue # Adjust count
            if not isinstance(labels_cpu, torch.Tensor):
                 if isinstance(labels_cpu, np.ndarray):
                     labels_cpu = torch.from_numpy(labels_cpu)
                 else:
                     print(f"Warning: Label batch type {type(labels_cpu)} not tensor or numpy. Skipping batch {i}.")
                     total_train -= B; continue # Adjust count

            # --- *** ISOLATED .to(device) CALLS *** ---
            try:
                # Step 1: Move labels
                labels = labels_cpu.to(device, non_blocking=pin_memory)

                # Step 2: Move inputs
                inputs = inputs_cpu.to(device, non_blocking=pin_memory)

            except RuntimeError as e_move:
                print(f"\n!!! RUNTIME ERROR during .to(device) in Batch {i}: {e_move}")
                print(f"    Occurred trying to move tensor after loading batch.")
                print(f"    Inputs CPU shape: {inputs_cpu.shape}, dtype: {inputs_cpu.dtype}")
                print(f"    Labels CPU shape: {labels_cpu.shape}, dtype: {labels_cpu.dtype}")
                print(f"    torch.cuda.is_available(): {torch.cuda.is_available()}")
                print(f"    torch.cuda.current_device(): {torch.cuda.current_device()}")
                raise e_move
            except Exception as e_other:
                 print(f"\n!!! UNEXPECTED ERROR during .to(device) in Batch {i}: {e_other}")
                 raise e_other
            # --- *** END ISOLATED .to(device) CALLS *** ---

            # --- Input Shape Handling (operate on GPU tensors now) ---
            if inputs.dim() < 4: # Need at least B, T, C, H or B, T, ...
                 print(f"\nError: Train incorrect input dims: {inputs.dim()}. Expected >= 4. Skipping batch {i}.")
                 total_train -= B; continue

            if time_dim_idx == 1: # Input is [B, T, ...]
                inputs = inputs.permute(1, 0, *inputs.shape[2:]).contiguous()
            elif time_dim_idx != 0: # Input is not [T, B, ...] or [B, T, ...] or unknown
                 print(f"\nError: Train Unexpected time_dim_idx: {time_dim_idx}. Skipping batch {i}.")
                 total_train -= B; continue

            if inputs.dim() != 5 or inputs.shape[0] != T:
                 print(f"\nError: Incorrect input shape before resize: {inputs.shape}. Expected [T={T}, B, C, H, W]. Skipping batch {i}.")
                 total_train -= B; continue

            # --- Apply Resizing Step (on GPU tensors) ---
            try:
                t_dim, b_dim, c_dim, h_dim, w_dim = inputs.shape
                # Ensure input is float before resize if needed (Resize usually handles it)
                # inputs = inputs.float()
                inputs_reshaped = inputs.view(t_dim * b_dim, c_dim, h_dim, w_dim)
                inputs_resized = resizer(inputs_reshaped) # Apply to Tensor B*T, C, H, W
                inputs = inputs_resized.view(t_dim, b_dim, c_dim, 224, 224) # Reshape back
            except Exception as e_resize:
                 print(f"\nError during resize in train loop: {e_resize}. Input dtype: {inputs.dtype}, shape: {inputs.shape}. Skipping batch {i}.")
                 total_train -= B # Adjust count as this batch won't proceed
                 continue
            # --- End Resizing Step ---

            # --- Training Step ---
            functional.reset_net(model)
            optimizer.zero_grad(set_to_none=True)

            batch_out_seq = []
            # Apply AMP context to the forward pass simulation
            with autocast(enabled=use_amp):
                for t in range(T):
                    input_t = inputs[t] # Get input for timestep t [B, C, 224, 224]
                    # Ensure input_t is float for model if necessary
                    # input_t = input_t.float()
                    out = model(input_t) # Model expects [B, C, H, W]
                    batch_out_seq.append(out)

            if not batch_out_seq: continue # Should not happen if input checks pass
            out_seq = torch.stack(batch_out_seq, dim=0) # [T, B, N]

            # Calculate Loss using TTFS Logits (still within AMP context)
            with autocast(enabled=use_amp):
                first_spikes = get_first_spike_times(out_seq, T)
                if first_spikes is None:
                    print(f"Warning: get_first_spike_times returned None for batch {i}. Skipping loss calculation.")
                    continue # Skip batch if spike times invalid

                logits = -first_spikes.float() # Ensure float
                loss = criterion(logits, labels) # labels are already on device

            # Backward Pass & Optimizer Step (using AMP scaler)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Track Metrics
            running_loss += loss.item()
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct_train += (preds == labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Train Acc': f"{100.0*correct_train/total_train:.2f}%" if total_train > 0 else "0.00%"
             })
            # --- End Batch Loop ---

        # --- End of Epoch ---
        end_epoch_time = time.time()
        epoch_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
        epoch_acc = 100.0 * correct_train / total_train if total_train > 0 else 0
        print(f"Epoch {epoch+1} Summary | Time: {end_epoch_time - start_epoch_time:.2f}s | Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

        # --- Validation Step ---
        val_accuracy, avg_val_spikes = evaluate_ttfs(model, test_loader, T, device, time_dim_idx, plot_first_batch=(epoch == epochs-1))
        if val_accuracy >= 0: # Check if accuracy was calculated
             print(f"Epoch {epoch+1} Validation Accuracy: {val_accuracy:.2f}% | Avg Spikes: {avg_val_spikes:.4f}")
             # Save Best Model
             if val_accuracy > best_val_acc:
                 best_val_acc = val_accuracy
                 print(f"Saving best model checkpoint to: {best_model_save_path} (Acc: {best_val_acc:.2f}%)")
                 torch.save(model.state_dict(), best_model_save_path)
        else:
             print(f"Epoch {epoch+1} Validation: Could not calculate accuracy.")
             # Save checkpoint anyway based on epoch completion
             e_save_path = model_save_path.replace('.pth', f'_e{epoch+1}.pth')
             print(f"Saving epoch checkpoint to: {e_save_path}")
             torch.save(model.state_dict(), e_save_path)
        # --- End Epoch Loop ---

    print("\nTraining finished.")
    # Save the final model state from the last epoch
    print(f"Saving final epoch model to: {model_save_path}")
    torch.save(model.state_dict(), model_save_path)


###############################################
# Evaluation Function (Updated with Resizing and Isolated .to(device))
###############################################
def evaluate_ttfs(model, test_loader, T, device, time_dim_idx, plot_first_batch=True):
    model.eval()
    correct = 0
    total = 0
    spike_sum_total = 0.0
    # Calculate neuron count based on final layer output size
    try:
        # Assumes the model passed in has the classifier attribute correctly defined
        output_dim = model.classifier[-1].out_features if isinstance(model.classifier[-1], nn.Linear) else num_classes
    except AttributeError:
         print("Warning: Could not infer output_dim from model.classifier. Using num_classes config.")
         output_dim = num_classes # num_classes needs to be accessible (e.g., global or passed)
    except Exception as e:
        print(f"Warning: Error inferring output_dim ({e}). Using num_classes config.")
        output_dim = num_classes

    neuron_count_total = 0
    first_batch_out_seq_spikes = None
    first_batch_labels = None
    first_batch_preds = None
    resizer = transforms.Resize((224, 224), antialias=True) # Define resizer once

    start_eval_time = time.time()
    print(f"\nStarting TTFS Evaluation (T={T})...")

    with torch.no_grad():
        for i, batch_data in enumerate(tqdm(test_loader, desc="Evaluating SNN TTFS")):
            # --- Data Loading and Prep ---
            if isinstance(batch_data, (list, tuple)):
                inputs_cpu, labels_cpu = batch_data[0], batch_data[1]
            else: continue

            B = labels_cpu.size(0) if isinstance(labels_cpu, torch.Tensor) else 0
            if B == 0: continue
            total += B

            if not isinstance(inputs_cpu, torch.Tensor):
                 if isinstance(inputs_cpu, np.ndarray): inputs_cpu = torch.from_numpy(inputs_cpu)
                 else: total -= B; continue # Skip if not tensor/numpy
            if not isinstance(labels_cpu, torch.Tensor):
                 if isinstance(labels_cpu, np.ndarray): labels_cpu = torch.from_numpy(labels_cpu)
                 else: total -= B; continue # Skip if not tensor/numpy


            # --- *** ISOLATED .to(device) CALLS *** ---
            try:
                labels = labels_cpu.to(device, non_blocking=pin_memory)
                inputs = inputs_cpu.to(device, non_blocking=pin_memory)
            except RuntimeError as e_move:
                print(f"\n!!! RUNTIME ERROR during .to(device) in Eval Batch {i}: {e_move}")
                print(f"    Inputs CPU shape: {inputs_cpu.shape}, dtype: {inputs_cpu.dtype}")
                print(f"    Labels CPU shape: {labels_cpu.shape}, dtype: {labels_cpu.dtype}")
                # Skip rest of eval? Or just this batch? Let's skip batch.
                total -= B; continue
            except Exception as e_other:
                 print(f"\n!!! UNEXPECTED ERROR during .to(device) in Eval Batch {i}: {e_other}")
                 total -= B; continue
            # --- *** END ISOLATED .to(device) CALLS *** ---

            # --- Input Shape Handling ---
            if inputs.dim() < 4:
                 print(f"\nError: Eval incorrect input dims: {inputs.dim()}. Expected >= 4. Skipping batch {i}.")
                 total -= B; continue
            if time_dim_idx == 1: # Input is [B, T, ...]
                inputs = inputs.permute(1, 0, *inputs.shape[2:]).contiguous()
            elif time_dim_idx != 0:
                 print(f"\nError: Eval Unexpected time_dim_idx: {time_dim_idx}. Skipping batch {i}.")
                 total -= B; continue

            if inputs.dim() != 5 or inputs.shape[0] != T:
                 print(f"\nError: Eval incorrect input shape before resize: {inputs.shape}. Expected [T={T}, B, C, H, W]. Skipping batch {i}.")
                 total -= B; continue

            # --- Apply Resizing Step ---
            try:
                t_dim, b_dim, c_dim, h_dim, w_dim = inputs.shape
                # inputs = inputs.float() # Ensure float for resize if needed
                inputs_reshaped = inputs.view(t_dim * b_dim, c_dim, h_dim, w_dim)
                inputs_resized = resizer(inputs_reshaped)
                inputs = inputs_resized.view(t_dim, b_dim, c_dim, 224, 224)
            except Exception as e_resize:
                 print(f"\nError during resize in eval loop: {e_resize}. Input dtype: {inputs.dtype}, shape: {inputs.shape}. Skipping batch {i}.")
                 total -= B; continue
            # --- End Resizing Step ---

            # --- Evaluation Step ---
            functional.reset_net(model)
            batch_out_seq = []
            # No need for autocast during eval unless specific layers require it
            for t in range(T):
                input_t = inputs[t] # [B, C, 224, 224]
                # input_t = input_t.float() # Ensure float if model expects it
                out = model(input_t)
                batch_out_seq.append(out)

            if not batch_out_seq: continue
            batch_out_seq = torch.stack(batch_out_seq, dim=0) # [T, B, N]

            # Calculate TTFS prediction
            first_spikes = get_first_spike_times(batch_out_seq, T)
            if first_spikes is None: continue

            logits = -first_spikes.float()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

            # Collect stats
            batch_spikes = (batch_out_seq > 0).float()
            spike_sum_total += batch_spikes.sum().item()
            neuron_count_total += B * output_dim

            # Store data for visualization (only first batch)
            if i == 0:
                first_batch_out_seq_spikes = batch_spikes.cpu()
                first_batch_labels = labels.cpu()
                first_batch_preds = preds.cpu()
            # --- End Batch Loop ---

    # --- Final Metrics Calculation & Visualization ---
    end_eval_time = time.time()
    acc = 100.0 * correct / total if total > 0 else 0.0
    avg_spikes_per_step = spike_sum_total / (neuron_count_total * T) if neuron_count_total > 0 and T > 0 else 0.0

    print(f"\nEvaluation finished in {end_eval_time - start_eval_time:.2f} seconds.")
    if total > 0:
        print(f"Test Accuracy: {acc:.2f}% ({correct}/{total})")
        print(f"Avg. Spikes per Output Neuron per Timestep: {avg_spikes_per_step:.4f}")
    else:
        print("Test Accuracy: N/A (No samples processed successfully)")
        print(f"Avg. Spikes per Output Neuron per Timestep: N/A")


    # Visualization on the first batch:
    if plot_first_batch and first_batch_out_seq_spikes is not None:
        print("Generating visualizations for the first batch...")
        num_samples_to_plot = min(4, first_batch_out_seq_spikes.shape[1])
        for sample_idx in range(num_samples_to_plot):
             if sample_idx < len(first_batch_labels) and sample_idx < len(first_batch_preds):
                 plot_spike_raster(first_batch_out_seq_spikes,
                                   gt_label=first_batch_labels[sample_idx].item(),
                                   pred_label=first_batch_preds[sample_idx].item(),
                                   sample_idx=sample_idx,
                                   title_prefix=f"Eval_")
                 spike_counts_sample = first_batch_out_seq_spikes[:, sample_idx].sum(dim=0).numpy()
                 plot_spike_count_hist(spike_counts_sample,
                                       title=f"Eval Spike Count Hist (Sample {sample_idx}, GT={first_batch_labels[sample_idx].item()})")
             else:
                  print(f"Skipping plot for sample {sample_idx}, index out of bounds.")
    elif plot_first_batch:
         print("Could not generate visualizations: No evaluation data recorded for first batch.")

    return acc, avg_spikes_per_step




###############################################
# Main Execution Logic
###############################################
if __name__ == '__main__':
    # --- Instantiate SNN Model and Convert ---
    print("Loading ANN and converting to SNN...")
    ann_model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
    ann_model.classifier[-1] = nn.Linear(ann_model.classifier[-1].in_features, num_classes)
    snn_model = convert_ann_to_snn(ann_model).to(device)
    print(f"SNN model created using conversion method and moved to {device}.")

    # --- Data Loaders should be defined by now ---
    if 'train_loader' not in locals() or 'test_loader' not in locals():
        print("Error: DataLoaders not defined. Exiting.")
        exit()
    if time_dim_index == -1:
         print("Error: Could not determine time dimension index from data loader. Exiting.")
         # Or set a default, e.g., time_dim_index = 1 (B, T, ...)
         # time_dim_index = 1 # Set default manually if needed, check Tonic docs!
         exit()

    # --- Run Training or Evaluation ---
    do_training = True # Set to False to only evaluate a loaded model
    
    # --- Manual Batch Load Test ---
    print("\n--- Attempting Manual Batch Load and Move Test ---")
    try:
        manual_iter = iter(train_loader)
        m_inputs, m_labels = next(manual_iter)
        print(f"Manual batch loaded.")
        print(f"  Inputs type: {type(m_inputs)}, Labels type: {type(m_labels)}")

        # Convert if NumPy (shouldn't be needed if Tonic yields tensors, but check)
        if not isinstance(m_inputs, torch.Tensor):
            print("  Converting manual inputs from numpy...")
            m_inputs = torch.from_numpy(m_inputs)
        if not isinstance(m_labels, torch.Tensor):
            print("  Converting manual labels from numpy...")
            m_labels = torch.from_numpy(m_labels)

        print(f"  Inputs shape: {m_inputs.shape}, dtype: {m_inputs.dtype}")
        print(f"  Labels shape: {m_labels.shape}, dtype: {m_labels.dtype}")

        print(f"  Moving manual labels to {device}...")
        m_labels_gpu = m_labels.to(device)
        print(f"  Manual labels moved. Device: {m_labels_gpu.device}")

        print(f"  Moving manual inputs to {device}...")
        m_inputs_gpu = m_inputs.to(device)
        print(f"  Manual inputs moved. Device: {m_inputs_gpu.device}")

        print("--- Manual batch move test SUCCEEDED. ---")
        del manual_iter, m_inputs, m_labels, m_inputs_gpu, m_labels_gpu # Clean up
        torch.cuda.empty_cache() # Clear any memory used

    except StopIteration:
        print("--- Manual batch move test FAILED: DataLoader yielded no data. ---")
    except Exception as e_manual:
        print(f"--- Manual batch move test FAILED: {e_manual} ---")
        import traceback
        traceback.print_exc()
        print("--- Exiting due to failure in manual test ---")
        exit() # Stop if the basic loading/moving fails here

    if do_training:
        print("\nStarting TTFS Fine-Tuning on N-CIFAR10...")
        train_ttfs(snn_model, train_loader, test_loader, device, T, finetune_epochs, learning_rate, model_save_path, best_model_save_path, time_dim_index)
        # After training, load the best model saved during training for final evaluation
        if os.path.exists(best_model_save_path):
            print("\nLoading BEST model saved during training for final evaluation...")
            snn_model.load_state_dict(torch.load(best_model_save_path, map_location=device))
        else:
             print("\nNo best model found, evaluating model from final epoch saved at:", model_save_path)
             if os.path.exists(model_save_path):
                 snn_model.load_state_dict(torch.load(model_save_path, map_location=device))
             else:
                 print("Warning: Final epoch model also not found. Evaluating potentially untrained model state.")

    else: # Only evaluate
         loaded = False
         if os.path.exists(best_model_save_path):
             print(f"\nLoading BEST fine-tuned TTFS SNN model from: {best_model_save_path}")
             snn_model.load_state_dict(torch.load(best_model_save_path, map_location=device))
             loaded = True
         elif os.path.exists(model_save_path):
             print(f"\nLoading FINAL epoch fine-tuned TTFS SNN model from: {model_save_path}")
             snn_model.load_state_dict(torch.load(model_save_path, map_location=device))
             loaded = True
         else:
              print("\nNo fine-tuned model found to evaluate. Evaluating initial converted model.")
         if loaded: print("Loaded existing model.")


    # --- Final Evaluation ---
    print("\n--- Final Evaluation ---")
    evaluate_ttfs(snn_model, test_loader, T, device, time_dim_index, plot_first_batch=True)

    print("\nScript finished.")
    