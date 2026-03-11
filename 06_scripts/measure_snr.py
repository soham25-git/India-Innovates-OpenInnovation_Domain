from hailo_sdk_client import ClientRunner
import numpy as np
import os

def calculate_snr(har_path):
    print(f"Calculating SNR for {os.path.basename(har_path)}...")
    runner = ClientRunner()
    runner.load_har(har_path)
    # The statistics are already collected during optimization in the HAR
    snr_results = runner.get_snr()
    
    # get_snr returns a dict of layer_name -> snr_value
    # We'll take the minimum SNR (worst case) or average of output layers
    if not snr_results:
        return "N/A"
    
    # Filter for output layers or just get the minimum across the network
    # For a quick summary, we'll return the average SNR across all layers
    snr_values = list(snr_results.values())
    avg_snr = sum(snr_values) / len(snr_values)
    min_snr = min(snr_values)
    
    return avg_snr, min_snr

models = [
    "/home/soham/Project/HAILO/output/yolov8n_seg_quantized.har",
    "/home/soham/Project/HAILO/output/potato_classifier_quantized.har",
    "/home/soham/Project/HAILO/output/lesion_segmentation_quantized.har"
]

print("-" * 50)
print(f"{'Model':<30} | {'Avg SNR':<10} | {'Min SNR':<10}")
print("-" * 50)

for m in models:
    try:
        avg, low = calculate_snr(m)
        print(f"{os.path.basename(m):<30} | {avg:>10.2f} | {low:>10.2f}")
    except Exception as e:
        print(f"{os.path.basename(m):<30} | Failed: {e}")
