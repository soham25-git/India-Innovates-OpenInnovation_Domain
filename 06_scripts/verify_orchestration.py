import sys
import os
from orchestrator import PotatoPipeline

def verify_logic():
    print("--- Verifying Orchestration Logic (Mock) ---")
    config_path = ".gsd/hailo_pipeline_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config not found at {config_path}")
        return False
        
    try:
        # Initialize pipeline in mock mode
        pipeline = PotatoPipeline(config_path, mock=True)
        
        # Test Stage 1: Leaf Detection
        dummy_frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        import numpy as np # Local import for dummy
        
        leaf_rois = pipeline.stage_1_leaf_segmentation(dummy_frame)
        print(f"✅ Stage 1: Found {len(leaf_rois)} mock leaves.")
        
        # Test Stage 2: Analysis
        results = pipeline.stage_2_analysis(dummy_frame, leaf_rois)
        print(f"✅ Stage 2: Analyzed {len(results)} plants.")
        
        # Verify result structure
        if len(results) > 0 and 'class' in results[0]:
            print("✅ Data flow verified: Case -> ROI -> Analysis.")
            return True
        else:
            print("❌ Data flow incomplete.")
            return False
            
    except Exception as e:
        print(f"❌ Logic Verification Failed: {e}")
        return False

if __name__ == "__main__":
    import numpy as np
    if not verify_logic():
        sys.exit(1)
