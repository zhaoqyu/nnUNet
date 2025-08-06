import os
import nibabel as nib
import json
from pathlib import Path
import shutil


import argparse
import multiprocessing
import shutil
from typing import Optional
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.paths import nnUNet_raw
from nnunetv2.utilities.dataset_name_id_conversion import find_candidate_datasets
from nnunetv2.configuration import default_num_processes
import numpy as np




# 🛠 CONFIGURATION – Update these paths
dataset_root = Path("/home/qingyu/datasets/ds004199-1.0.6")  # BIDS dataset folder
output_dir = Path("nnUNet_raw/Dataset999_ds004199")

# Create output directories
imagesTr = output_dir / "imagesTr"
labelsTr = output_dir / "labelsTr"
imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)

def list_subjects():
    return sorted([d for d in dataset_root.glob("sub-*") if d.is_dir()])

def find_modalities(subj_path):
    anat = list(subj_path.glob("anat/*T1w.nii*"))
    seg = list(subj_path.glob("anat/*FLAIR*.nii*"))  # catch *_seg.nii.gz
    return anat, seg

def copy_and_rename(src, dst):
    shutil.copy(src, dst)

def convert():
    subjects = list_subjects()
    print(f"Found {len(subjects)} subjects.")  # ds004199 has ~170 cases :contentReference[oaicite:1]{index=1}

    training_list = []
    for idx, subj in enumerate(subjects):
        anat_files, seg_files = find_modalities(subj)
        if not anat_files or not seg_files:
            print(f"⚠️ Skipping {subj.name}: missing modality or segmentation.")
            continue

        case_id = f"case_{idx:05d}"
        img_src = anat_files[0]
        seg_src = seg_files[0]

        img_dst = imagesTr / f"{case_id}_000.nii.gz"
        seg_dst = labelsTr / f"{case_id}.nii.gz"
        copy_and_rename(img_src, img_dst)
        copy_and_rename(seg_src, seg_dst)

        training_list.append({
            "image": f"./imagesTr/{img_dst.name}",
            "label": f"./labelsTr/{seg_dst.name}"
        })

    # 🎯 Create dataset.json
    dataset_json = {
        "name": "ds004199",
        "description": "FCD Type II epilepsy pre‑surgery MRI (T1w + segmentation)",
        "tensorImageSize": "3D",
        "reference": "doi:10.18112/openneuro.ds004199.v1.0.6",
        "licence": "CC0",
        "modality": {"0": "T1"},
        "labels": {"0": "background", "1": "lesion"},
        "numTraining": len(training_list),
        "numTest": 0,
        "training": training_list,
        "test": []
    }


    dataset_json = load_json(dataset_json)
    dataset_json['labels'] = {j: int(i) for i, j in dataset_json['labels'].items()}
    dataset_json['file_ending'] = ".nii.gz"
    dataset_json["channel_names"] = dataset_json["modality"]
    del dataset_json["modality"]
    del dataset_json["training"]
    del dataset_json["test"]
    save_json(dataset_json, join(nnUNet_raw, target_dataset_name, 'dataset.json'), sort_keys=False)


# if __name__ == "__main__":
#     convert()

def entry_point():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True,
                        help='Downloaded and extracted MSD dataset folder. CANNOT be nnUNetv1 dataset! Example: '
                             '/home/fabian/Downloads/Task05_Prostate')
    parser.add_argument('-overwrite_id', type=int, required=False, default=None,
                        help='Overwrite the dataset id. If not set we use the id of the MSD task (inferred from '
                             'folder name). Only use this if you already have an equivalently numbered dataset!')
    parser.add_argument('-np', type=int, required=False, default=default_num_processes,
                        help=f'Number of processes used. Default: {default_num_processes}')
    args = parser.parse_args()
    convert_msd_dataset(args.i, args.overwrite_id, args.np)


if __name__ == '__main__':
    convert_msd_dataset('/home/fabian/Downloads/Task05_Prostate', overwrite_target_id=201)



