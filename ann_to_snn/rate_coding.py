import os
import torch
import torch.nn as nn
import torch.optim as optim # Added for optimizer
import torchvision
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights
import spikingjelly.clock_driven.ann2snn as ann2snn
from spikingjelly.clock_driven import functional # For resetting network state
from tqdm import tqdm # For progress bar
import time # For timing

# -------------------------------
# Configuration
# -------------------------------
# --- Fine-Tuning Params ---
finetune_snn = True # Set to True to enable fine-tuning (or loading if already done)
finetune_epochs = 1 # Number of epochs for fine-tuning
learning_rate = 1e-5 # Learning rate for fine-tuning optimizer
train_finetune_timesteps = 25 # Timesteps used DURING fine-tuning per batch

# --- Evaluation Params ---
evaluation_timesteps = 50 # Fixed timesteps for FINAL evaluation

# --- Data/Model Params ---
num_classes = 10
batch_size = 4 # Adjust based on GPU memory
data_dir = './ann_to_snn/data/CIFAR_10'
ann_model_name = "vgg16" # Just for reference
# << NEW: Path for the *fine-tuned* SNN model >>
finetuned_model_save_path = f"./ann_to_snn/data/models/finetuned_{ann_model_name}_snn_cifar{num_classes}_e{finetune_epochs}_t{train_finetune_timesteps}.pth"
os.makedirs(os.path.dirname(finetuned_model_save_path), exist_ok=True) # Ensure directory exists

# -------------------------------
# Device Setup (CUDA / CPU only)
# -------------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using NVIDIA CUDA device.")
    # Optional: Clear cache if needed
    # torch.cuda.empty_cache()
else:
    device = torch.device("cpu")
    print("Using CPU device.")

print("PyTorch Version:", torch.__version__)
# Adjust num_workers based on the device and OS capabilities
# On Windows, num_workers > 0 can sometimes cause issues. Start with 0 or 2.
# On Linux with CUDA, 4 or 8 might be good starting points.
num_workers = 8 if device.type == 'cuda' else 0
pin_memory = False if device.type == 'cuda' else False
print(f"Using Device: {device}, Num Workers: {num_workers}, Pin Memory: {pin_memory}")


# -------------------------------
# Load and Modify Pre-trained VGG-16 for CIFAR-10
# -------------------------------
print("Loading pre-trained VGG-16 model...")
# Ensure the model is loaded onto the CPU first if memory is tight, then move to GPU
ann_model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
in_features = ann_model.classifier[-1].in_features
ann_model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)
# Keep ANN on CPU for conversion if GPU memory is a concern, then move SNN
# ann_model.to(device) # Moved later if needed for ANN eval
print("ANN model loaded.")

# -------------------------------
# DataLoaders for CIFAR-10 (Train and Test)
# -------------------------------
print("Setting up CIFAR-10 DataLoaders...")
# Correct ImageNet normalization for VGG16
transform = transforms.Compose([
    transforms.Resize(224), # VGG expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# --- Training Set (Needed for Fine-tuning) ---
train_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, # Shuffle is important for training
    pin_memory=pin_memory, num_workers=num_workers
)
print(f"Train DataLoader ready (Batch Size: {batch_size}).")

# --- Test Set ---
test_dataset = torchvision.datasets.CIFAR10(
    root=data_dir, train=False, download=True, transform=transform
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    pin_memory=pin_memory, num_workers=num_workers
)
print(f"Test DataLoader ready (Batch Size: {batch_size}).")


# --------------------------------------------------
# Optional: Evaluate Original ANN (Good for Baseline)
# --------------------------------------------------
print("\nEvaluating original ANN model accuracy...")
ann_model.to(device) # Move ANN to device for evaluation
ann_model.eval()
ann_correct = 0
ann_total = 0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating ANN"):
        images = images.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)
        outputs = ann_model(images)
        _, predicted = torch.max(outputs.data, 1)
        ann_total += labels.size(0)
        ann_correct += (predicted == labels).sum().item()

ann_accuracy = 100.0 * ann_correct / ann_total if ann_total > 0 else 0.0
print(f"Original ANN Accuracy on Test Set: {ann_accuracy:.2f}%")
if ann_accuracy < 80: # Adjusted threshold
    print("\nWARNING: Original ANN accuracy seems low. Conversion might not perform well.")
print("-" * 50)
# Optional: Move ANN back to CPU if GPU memory is very limited
# ann_model.to('cpu')
# torch.cuda.empty_cache() # If you moved it back

# -------------------------------
# ANN to SNN Conversion
# -------------------------------
print("Preparing for ANN to SNN conversion...")
# Use the test_loader for conversion normalization statistics (common practice)
# Keep ANN on CPU during conversion if memory is tight
converter = ann2snn.Converter(mode='max', dataloader=test_loader)
print("Converting ANN to SNN structure...")
snn_model = converter(ann_model)
print("SNN structure created.")
# It's crucial to move the SNN model to the target device *before* loading state_dict or training
snn_model.to(device)
print(f"SNN model moved to: {device}")


# --------------------------------------------------
# SNN Fine-Tuning (Train or Load Pre-trained)
# --------------------------------------------------
if finetune_snn:
    if os.path.exists(finetuned_model_save_path):
        print(f"\nLoading pre-existing fine-tuned SNN model from: {finetuned_model_save_path}")
        # Load the state dict onto the correct device
        snn_model.load_state_dict(torch.load(finetuned_model_save_path, map_location=device))
        print("Fine-tuned SNN model loaded.")

    else:
        print(f"\nStarting SNN Fine-tuning for {finetune_epochs} epochs...")
        print(f"  Training timesteps (T_train): {train_finetune_timesteps}")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Saving fine-tuned model to: {finetuned_model_save_path}")

        snn_model.train() # Set model to training mode

        # --- Optimizer and Loss ---
        # Use AdamW which often works well
        optimizer = optim.AdamW(snn_model.parameters(), lr=learning_rate)
        # Standard cross-entropy loss, requires averaging SNN output over time
        criterion = nn.CrossEntropyLoss()

        start_time = time.time()
        for epoch in range(finetune_epochs):
            print(f"\nEpoch {epoch+1}/{finetune_epochs}")
            running_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            snn_model.train() # Ensure model is in train mode each epoch

            # Use tqdm for progress bar over training batches
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")
            for i, (inputs, labels) in pbar:
                inputs = inputs.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                batch_size_current = inputs.size(0) # Get current batch size

                # Zero the parameter gradients
                optimizer.zero_grad()

                # --- SNN Simulation during Training ---
                functional.reset_net(snn_model) # Reset state for each batch!
                # Accumulate output over T_train timesteps
                # Use membrane potential or spike counts - here using output directly (often spike counts/rates)
                output_sum = torch.zeros(batch_size_current, num_classes, device=device)
                for t in range(train_finetune_timesteps):
                    output_step = snn_model(inputs) # Shape: (batch_size, num_classes)
                    output_sum += output_step

                # Average output over timesteps to get rate-like activation for loss
                output_avg = output_sum / train_finetune_timesteps

                # Calculate loss
                loss = criterion(output_avg, labels)

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # --- Track Stats ---
                running_loss += loss.item() * batch_size_current # Accumulate loss correctly
                _, predicted = torch.max(output_avg.data, 1)
                epoch_total += batch_size_current
                epoch_correct += (predicted == labels).sum().item()

                # Update progress bar description
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Avg Loss': f"{running_loss / epoch_total:.4f}",
                    'Acc': f"{100*epoch_correct/epoch_total:.2f}%"
                 })
                # --- End Batch ---

            epoch_loss = running_loss / len(train_dataset)
            epoch_acc = 100.0 * epoch_correct / epoch_total
            print(f"Epoch {epoch+1} Summary: Average Loss: {epoch_loss:.6f}, Accuracy: {epoch_acc:.2f}%")

            # Optional: Add validation loop here if needed

        end_time = time.time()
        print(f"\nFine-tuning finished in {end_time - start_time:.2f} seconds.")

        # --- Save the Fine-tuned Model ---
        print(f"Saving fine-tuned SNN model to: {finetuned_model_save_path}")
        torch.save(snn_model.state_dict(), finetuned_model_save_path)
        print("Model saved.")

else:
    print("\nSNN fine-tuning is disabled.")

print("-" * 50)

# --------------------------------------------------
# Final SNN Evaluation (Using the loaded or newly fine-tuned model)
# --------------------------------------------------
print(f"\nStarting FINAL SNN Evaluation...")
print(f"  Using T = {evaluation_timesteps} timesteps for evaluation.")

snn_model.eval() # Set model to evaluation mode IMPORTANT!
correct_predictions = 0
total_samples = 0

start_eval_time = time.time()
with torch.no_grad():
    # Use tqdm for progress bar over test batches
    for inputs, labels in tqdm(test_loader, desc=f"Evaluating SNN (T={evaluation_timesteps})"):
        inputs = inputs.to(device, non_blocking=pin_memory)
        labels = labels.to(device, non_blocking=pin_memory)
        batch_size_current = inputs.size(0)
        total_samples += batch_size_current

        # --- SNN Simulation for Evaluation ---
        functional.reset_net(snn_model) # Reset state before each batch

        # Accumulate spikes over evaluation_timesteps
        # Shape: (batch_size, num_classes)
        spike_counts = torch.zeros(batch_size_current, num_classes, device=device)
        for t in range(evaluation_timesteps):
            output_step = snn_model(inputs)
            spike_counts += output_step # accumulate spike counts (or rate if output is rate)

        # Decode prediction: highest spike count neuron
        # Note: If snn_model outputs avg firing rate, argmax still applies
        predictions = spike_counts.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()

end_eval_time = time.time()
print(f"\nEvaluation finished in {end_eval_time - start_eval_time:.2f} seconds.")

# --- Calculate and Print Final Accuracy ---
final_accuracy = 100.0 * correct_predictions / total_samples if total_samples > 0 else 0.0
print("\n" + "="*50)
print("=== Final SNN Model Evaluation Result ===")
print("="*50)
print(f"Timesteps (T) used for evaluation: {evaluation_timesteps}")
if finetune_snn:
    print(f"Model source: {'Loaded from ' + finetuned_model_save_path if os.path.exists(finetuned_model_save_path) else 'Newly fine-tuned'}")
    print(f"(Fine-tuning was done with {finetune_epochs} epochs, T_train={train_finetune_timesteps}, LR={learning_rate})")
else:
     print(f"Model source: Converted ANN (No fine-tuning performed)")
print(f"\nFinal SNN Accuracy on Test Set: {final_accuracy:.2f}%")
print("="*50)