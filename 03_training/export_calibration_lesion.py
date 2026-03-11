import numpy as np
import cv2
import argparse
from pathlib import Path

def prepare_calib_lesion(images_dir, output_file, num_samples=1000):
    images_list = list(Path(images_dir).glob('*.jpg')) + \
                  list(Path(images_dir).glob('*.png')) + \
                  list(Path(images_dir).glob('*.jpeg'))
    np.random.shuffle(images_list)
    images_list = images_list[:num_samples]

    data = []
    print(f"Processing {len(images_list)} images for Lesion Segmentation calibration...")
    for i, img_path in enumerate(images_list):
        img = cv2.imread(str(img_path))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # UNet usually expects 256x256 simple resize
        img = cv2.resize(img, (256, 256))
        data.append(img)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(images_list)}")

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
    prepare_calib_lesion(args.images_dir, args.output, args.num_samples)
