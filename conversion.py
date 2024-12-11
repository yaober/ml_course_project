# Import Libraries
import os
import numpy as np
import pandas as pd
import pydicom
import numpy as np
# Convert DICOM to PNG
def convert_dicom_to_png_worker(args):
    dicom_path, output_dir = args
    try:
        dcm = pydicom.read_file(dicom_path)
        img = dcm.pixel_array.astype(np.float32)

        img_min = img.min()
        img_max = img.max()
        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

        img_id = os.path.basename(dicom_path).replace(".dcm", ".png")
        output_file = os.path.join(output_dir, img_id)
        Image.fromarray(img).save(output_file)
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")


# Batch Convert DICOM to PNG with Multi-Processing
def batch_convert_to_png_mp(dicom_dir, output_dir, num_workers=72):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dicom_files = [
        os.path.join(dicom_dir, file)
        for file in os.listdir(dicom_dir)
        if file.endswith(".dcm")
    ]


# Convert Images
batch_convert_to_png_mp(TRAIN_DICOM_DIR, TRAIN_PNG_DIR)

