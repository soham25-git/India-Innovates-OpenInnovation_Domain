import numpy as np
import cv2
import os
import argparse

def prepare_calib_classifier(images_dir, output_file, num_samples=1000):
    images_list = []
    for root, dirs, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                images_list.append(os.path.join(root, f))

    np.random.shuffle(images_list)
    images_list = images_list[:num_samples]

    data = []
    print(f"Processing {len(images_list)} images for Classifier calibration with CENTER CROP...")
    for i, img_path in enumerate(images_list):
        img = cv2.imread(img_path)
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Match classifier.py: resize side to 256, then center crop to 224
        h, w = img.shape[:2]
        scale = 256.0 / min(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        h, w = img.shape[:2]
        cy, cx = h // 2, w // 2
        img = img[cy-112:cy+112, cx-112:cx+112]
        
        data.append(img)
        if (i+1) % 100 == 0: print(f"  Processed {i+1}/{len(images_list)}")

    data = np.array(data, dtype=np.uint8)
    print(f"Calibration array shape: {data.shape}")
    np.save(output_file, data)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()
    prepare_calib_classifier(args.images_dir, args.output, args.num_samples)
