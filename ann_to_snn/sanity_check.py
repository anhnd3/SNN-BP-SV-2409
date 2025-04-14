import torch
import sys

print(f"--- Minimal CUDA Test ---")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    if torch.cuda.device_count() > 0:
        try:
            device_index = 0 # Test device 0 specifically
            device = torch.device(f"cuda:{device_index}")
            print(f"Attempting to use device: {device}")
            print(f"Device name: {torch.cuda.get_device_name(device_index)}")

            # Test 1: Create tensor on CPU and move to GPU
            print("\nTest 1: Moving CPU tensor to GPU...")
            cpu_tensor = torch.randn(5, 5)
            print(f"  CPU tensor device: {cpu_tensor.device}")
            gpu_tensor = cpu_tensor.to(device)
            print(f"  GPU tensor device: {gpu_tensor.device}")
            print("  Test 1 SUCCEEDED.")

            # Test 2: Create tensor directly on GPU
            print("\nTest 2: Creating tensor directly on GPU...")
            gpu_tensor_direct = torch.randn(5, 5, device=device)
            print(f"  Direct GPU tensor device: {gpu_tensor_direct.device}")
            print("  Test 2 SUCCEEDED.")

            # Test 3: Simple GPU operation
            print("\nTest 3: Performing simple GPU operation...")
            result_tensor = gpu_tensor + gpu_tensor_direct
            print(f"  Result tensor device: {result_tensor.device}")
            # Try moving result back to CPU
            _ = result_tensor.cpu()
            print("  Test 3 SUCCEEDED.")

            print("\n--- Minimal CUDA test completed successfully! ---")

        except Exception as e:
            print(f"\n!!! Error during minimal CUDA test: {e}")
            print("--- Minimal CUDA test FAILED ---")
            # Print traceback for detailed error
            import traceback
            traceback.print_exc()
    else:
        print("CUDA is available but device count is 0?")
else:
    print("CUDA not available to PyTorch.")

print("-" * 27)