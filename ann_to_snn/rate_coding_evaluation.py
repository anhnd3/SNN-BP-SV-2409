import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import VGG16_Weights

# Import SpikingJelly components needed for conversion and functional reset
import spikingjelly.clock_driven.ann2snn as ann2snn
from spikingjelly.clock_driven import functional # For resetting network state

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm # For progress bar

# -------------------------------
# CONFIGURATION FOR EVALUATION
# -------------------------------
# Evaluation Timesteps Sweep
time_steps_list = [5, 10, 20, 30, 40, 50, 60]

# Data and Model Paths (Using paths from previous context)
# IMPORTANT: Make sure data_dir points to where CIFAR-10 IS or WILL BE downloaded
data_dir = './ann_to_snn/data/CIFAR_10'
# Paths to the fine-tuned model state dicts saved from your training script
# Ensure these files exist
model_base_dir = './ann_to_snn/data/models'
model_path_T20 = os.path.join(model_base_dir, 'finetuned_vgg16_snn_cifar10_e1_t20.pth') # Verify filename if needed
model_path_T25 = os.path.join(model_base_dir, 'finetuned_vgg16_snn_cifar10_e1_t25.pth') # Verify filename if needed

results_dir = './ann_to_snn/data/results'
results_img_path = os.path.join(results_dir, "evaluation_ann2snn_rate_coding_charts.png")
os.makedirs(results_dir, exist_ok=True) # Ensure results directory exists

# Evaluation Hyperparameters
eval_batch_size = 16 # Batch size for evaluation (Adjust based on memory)
num_classes = 10
ann_model_name = "vgg16" # For reference

# Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# DataLoader Params (Adjust based on your system)
eval_num_workers = 8 if device.type == 'cuda' else 0 # Or set to 0 if issues persist
eval_pin_memory = False # Set based on preference/system
print(f"Evaluation DataLoader using Num Workers: {eval_num_workers}, Pin Memory: {eval_pin_memory}")

# =============================================================================
# Define Evaluation Function (Rate Coding)
# =============================================================================
def evaluate_model(model, T_eval, test_loader, device, num_classes):
    """
    Evaluate the SNN model using rate coding for the given evaluation timesteps.
    """
    model.eval() # Set model to evaluation mode
    correct_predictions = 0
    total_samples = 0
    total_eval_time = 0.0

    desc = f"Evaluating SNN (T_eval={T_eval})"
    with torch.no_grad(): # Disable gradients for evaluation
        for inputs, labels in tqdm(test_loader, desc=desc):
            # Move data to device *inside* the loop
            try:
                inputs = inputs.to(device, non_blocking=eval_pin_memory)
                labels = labels.to(device, non_blocking=eval_pin_memory)
            except RuntimeError as e:
                 print(f"\nCUDA Error moving batch to device: {e}. Skipping batch.")
                 continue # Skip this batch if moving fails

            batch_size_current = inputs.size(0)
            total_samples += batch_size_current

            # --- SNN Simulation for Evaluation ---
            start_batch_time = time.time()
            functional.reset_net(model) # Reset the network state

            # Accumulate output potential/spikes over T_eval steps
            output_sum = torch.zeros(batch_size_current, num_classes, device=device)
            try:
                for t in range(T_eval):
                    output_step = model(inputs) # Get output for one step: [B, N]
                    output_sum += output_step # Accumulate output signal
            except Exception as e_sim:
                print(f"\nError during SNN simulation step (t={t if 't' in locals() else 'unknown'}) : {e_sim}. Skipping batch.")
                total_samples -= batch_size_current # Adjust count
                continue # Skip to next batch

            # Decode prediction: highest accumulated signal wins
            predictions = output_sum.argmax(dim=1)
            correct_predictions += (predictions == labels).sum().item()
            end_batch_time = time.time()
            total_eval_time += (end_batch_time - start_batch_time)


    final_accuracy = (100.0 * correct_predictions / total_samples) if total_samples > 0 else 0.0
    # Calculate latency per sample (total eval time / num samples)
    avg_latency_per_sample = total_eval_time / total_samples if total_samples > 0 else 0.0

    print(f"\n  Evaluation for T_eval={T_eval} finished.")
    print(f"  Avg. Latency per Sample: {avg_latency_per_sample:.4f} seconds.")
    print(f"  Accuracy: {final_accuracy:.2f}% ({correct_predictions}/{total_samples})")

    return final_accuracy, avg_latency_per_sample

# =============================================================================
# Set up Test Dataset (CIFAR-10) and DataLoader with CORRECTED Transforms
# =============================================================================
print("\nSetting up CIFAR-10 Test DataLoader with Training Transforms...")
# <<< Use the SAME transforms as should have been used during training >>>
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # <<< Ensure Resize is present
    transforms.ToTensor(),
    normalize                  # <<< Ensure Normalize is present
])

try:
    # Ensure the root path is correct and writable/contains data
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True, # Download if not present
        transform=test_transforms # Use the corrected transforms
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=eval_num_workers,
        pin_memory=eval_pin_memory
    )
    print(f"Evaluation DataLoader ready. Found {len(test_dataset)} test samples.")
except Exception as e:
    print(f"Error creating test DataLoader: {e}")
    print(f"Please ensure CIFAR-10 dataset path ('{data_dir}') is correct and data exists/can be downloaded.")
    exit()


# =============================================================================
# Helper Function to Load ANN and Convert (with ANN moved to device first)
# THIS IS THE RE-CONVERSION APPROACH
# =============================================================================
def load_and_convert_ann(num_classes, device, test_loader_for_norm):
    """Loads ANN, adapts classifier, moves ANN to device, then converts to SNN structure."""
    print("Loading pre-trained VGG-16 ANN model...")
    # Load ANN initially on CPU
    ann_model = torchvision.models.vgg16(weights=VGG16_Weights.DEFAULT)
    in_features = ann_model.classifier[-1].in_features
    # Adapt classifier
    ann_model.classifier[-1] = nn.Linear(in_features=in_features, out_features=num_classes)
    print("ANN model loaded and classifier adapted.")

    # --- Move ANN to target device BEFORE conversion ---
    ann_model.to(device)
    print(f"ANN model moved to: {device} for conversion statistics")

    print("Preparing for ANN to SNN conversion (using GPU for ANN stats)...")
    # Converter will now use the GPU-based ann_model when iterating test_loader_for_norm
    # Ensure ann2snn is imported
    converter = ann2snn.Converter(mode='max', dataloader=test_loader_for_norm)
    print("Converting ANN to SNN structure...")
    snn_model_structure = converter(ann_model) # Convert the GPU-based ANN
    print("SNN structure created.")

    # Move the final SNN structure to the evaluation device
    snn_model_structure.to(device)
    print(f"SNN structure is on device: {device}")

    # Clean up ANN from GPU memory if no longer needed
    del ann_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return snn_model_structure # Return the SNN model structure (on device)


# =============================================================================
# Main Evaluation Sweep
# =============================================================================
if __name__ == '__main__': # Protect execution

    # Prepare lists for storing the results
    results = {
        "Model": [],
        "T_eval": [],
        "Accuracy (%)": [],
        "Latency (s/sample)": []
    }

    # Dictionary mapping model labels to their saved state_dict paths
    models_to_evaluate = {
        # "T_train=20": model_path_T20,
        "T_train=25": model_path_T25
    }

    # Ensure test_loader is defined before the loop
    if 'test_loader' not in locals():
        print("Error: test_loader not defined before evaluation loop.")
        exit()

    for model_label, model_path in models_to_evaluate.items():
        print(f"\n{'='*20} Evaluating Model: {model_label} {'='*20}")

        if not os.path.exists(model_path):
            print(f"Model state_dict file not found: {model_path}. Skipping evaluation.")
            continue

        snn_model = None # Initialize variable
        state_dict = None # Initialize variable
        # --- Instantiate SNN Structure via Re-conversion and Load State Dict ---
        try:
            # Step 1: Re-create the SNN structure using the same conversion process
            # Pass the actual test_loader used for evaluation also for conversion stats
            print("Re-creating SNN structure via ANN conversion...")
            snn_model = load_and_convert_ann(num_classes, device, test_loader)

            # Step 2: Load the fine-tuned weights into the created structure
            print(f"Loading fine-tuned state_dict from: {model_path}")
            state_dict = torch.load(model_path, map_location=device)

            # Optional: Fix potential state_dict key mismatch if saved from DataParallel
            # from collections import OrderedDict
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = k[7:] if k.startswith('module.') else k
            #     new_state_dict[name] = v
            # state_dict = new_state_dict

            snn_model.load_state_dict(state_dict)
            print("Fine-tuned SNN state_dict loaded successfully into converted structure.")

        except FileNotFoundError:
             print(f"ERROR: State dict file not found at {model_path} during load attempt.")
             if snn_model is not None: del snn_model
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             continue # Skip to next model
        except RuntimeError as e:
             print(f"ERROR: Failed to load state_dict for {model_label} from {model_path}: {e}")
             print("Architecture created by converter might mismatch saved state_dict or file corrupted. Skipping.")
             if snn_model is not None: del snn_model
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             continue # Skip to next model
        except Exception as e: # Catch any other instantiation/loading errors
             print(f"ERROR: An unexpected error occurred preparing {model_label}: {e}")
             if snn_model is not None: del snn_model
             if torch.cuda.is_available(): torch.cuda.empty_cache()
             continue # Skip to next model


        # --- Evaluate across different T_eval ---
        if snn_model is not None: # Proceed only if model loaded successfully
            for t_eval in time_steps_list:
                print(f"\n[{model_label}] Evaluating for T_eval = {t_eval}...")
                try:
                     # Ensure model is on the correct device before evaluation call
                     snn_model.to(device)
                     acc, lat = evaluate_model(snn_model, T_eval=t_eval, test_loader=test_loader, device=device, num_classes=num_classes)
                     results["Model"].append(model_label)
                     results["T_eval"].append(t_eval)
                     results["Accuracy (%)"].append(acc)
                     results["Latency (s/sample)"].append(lat)
                except Exception as e_eval:
                     print(f"ERROR during evaluation for T_eval={t_eval}: {e_eval}")
                     # Record failure or skip point
                     results["Model"].append(model_label)
                     results["T_eval"].append(t_eval)
                     results["Accuracy (%)"].append(float('nan')) # Indicate error
                     results["Latency (s/sample)"].append(float('nan'))
        else:
             print(f"Skipping evaluation loop for {model_label} as model preparation failed.")


        # --- Clean up GPU memory for the current model ---
        print(f"Finished evaluations for {model_label}.")
        if snn_model is not None: del snn_model
        if state_dict is not None: del state_dict
        if torch.cuda.is_available(): torch.cuda.empty_cache() # Clear cached memory
        print(f"Cleaned up memory for {model_label}.")


    # --- Display and Save Results ---
    if results["Model"]: # Check if any results were collected
        df_results = pd.DataFrame(results)
        print("\n--- Final Evaluation Results Table ---")
        # Display full dataframe
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df_results)
        # Save to CSV
        csv_path = os.path.join(results_dir, "evaluation_summary.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # --- Plotting ---
        print("\nGenerating plots...")
        try:
            # Ensure results dataframe is not empty before plotting
            if not df_results.empty:
                fig, axs = plt.subplots(1, 2, figsize=(16, 6)) # Slightly wider figure

                # Subplot 1: Accuracy vs. Evaluation timesteps
                sns.lineplot(data=df_results, x="T_eval", y="Accuracy (%)", hue="Model", marker='o', ax=axs[0])
                axs[0].set_xlabel("Evaluation Timesteps (T_eval)")
                axs[0].set_ylabel("Accuracy (%)")
                axs[0].set_title("Accuracy vs. Evaluation Timesteps")
                axs[0].grid(True, which="both", linestyle="--", alpha=0.6)
                axs[0].legend(title="Model (Trained with)")

                # Subplot 2: Latency vs. Evaluation timesteps
                sns.lineplot(data=df_results, x="T_eval", y="Latency (s/sample)", hue="Model", marker='o', ax=axs[1])
                axs[1].set_xlabel("Evaluation Timesteps (T_eval)")
                axs[1].set_ylabel("Avg. Latency per Sample (s)")
                axs[1].set_title("Latency vs. Evaluation Timesteps")
                axs[1].grid(True, which="both", linestyle="--", alpha=0.6)
                axs[1].legend(title="Model (Trained with)")

                plt.suptitle("SNN Rate Coding Evaluation: Accuracy and Latency Trade-off", fontsize=16)
                plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout
                plt.savefig(results_img_path)
                print(f"Plots saved to {results_img_path}")
                # plt.show() # Optionally display plot interactively
                plt.close() # Close the figure
            else:
                print("Skipping plot generation as no valid results were collected.")

        except Exception as e_plot:
            print(f"Error generating plots: {e_plot}")
            print("Ensure pandas, matplotlib, and seaborn are installed: pip install pandas matplotlib seaborn")

    else:
        print("\nNo evaluation results collected. Check model paths and evaluation process.")

    print("\nEvaluation script finished.")