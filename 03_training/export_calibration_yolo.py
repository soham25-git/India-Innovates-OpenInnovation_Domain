import numpy as np
import cv2
import argparse
from pathlib import Path

def letterbox(im, new_shape=(320, 320), color=(114, 114, 114)):
    # Standard YOLOv8 letterbox
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im

def prepare_calib_yolo(images_dir, output_file, num_samples=1000, size=320):
    images_list = list(Path(images_dir).glob('*.jpg')) + \
                  list(Path(images_dir).glob('*.png')) + \
                  list(Path(images_dir).glob('*.jpeg'))
    np.random.shuffle(images_list)
    images_list = images_list[:num_samples]

    data = []
    print(f"Processing {len(images_list)} images for YOLO calibration with LETTERBOX at {size}x{size}...")
    for i, img_path in enumerate(images_list):
        img = cv2.imread(str(img_path))
        if img is None: continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = letterbox(img, (size, size))
        data.append(img)
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(images_list)}")

    data = np.array(data, dtype=np.uint8)
    print(f"Calibration array shape: {data.shape}, dtype: {data.dtype}")
    np.save(output_file, data)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--size", type=int, default=320)
    args = parser.parse_args()
    prepare_calib_yolo(args.images_dir, args.output, args.num_samples, args.size)
