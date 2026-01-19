import time
import numpy as np
import torch
from ultralytics import YOLO

def measure_fps(device, model_path='yolov8n.pt', img_size=640, num_runs=50):
    print(f"\nTesting on {device.upper()}...")
    
    try:
        # Force device
        model = YOLO(model_path)
        
        # Dummy input for warming up
        dummy_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        # Warmup
        print("Warming up...")
        model.predict(dummy_img, device=device, verbose=False)
        model.predict(dummy_img, device=device, verbose=False)
        
        # Run
        print(f"Running {num_runs} iterations...")
        start_time = time.time()
        for _ in range(num_runs):
            model.predict(dummy_img, device=device, verbose=False)
        end_time = time.time()
        
        total_time = end_time - start_time
        fps = num_runs / total_time
        print(f"Result: {fps:.2f} FPS on {device.upper()}")
        return fps
        
    except Exception as e:
        print(f"Could not run on {device}: {e}")
        return 0

def main():
    print("--- Benchmark: CPU vs GPU ---")
    
    # 1. Test CPU
    cpu_fps = measure_fps('cpu')
    
    # 2. Test GPU (if available)
    if torch.cuda.is_available():
        gpu_fps = measure_fps('cuda') # or '0'
    else:
        print("\nNo GPU detected (CUDA). Skipping GPU test.")
        gpu_fps = 0
        
    # 3. Comparison
    print("\n--- Final Comparison ---")
    print(f"CPU Speed: {cpu_fps:.2f} FPS")
    if gpu_fps > 0:
        print(f"GPU Speed: {gpu_fps:.2f} FPS")
        print(f"Speedup: {gpu_fps/cpu_fps:.1f}x faster on GPU")
    else:
        print("GPU Speed: N/A")

if __name__ == "__main__":
    main()
