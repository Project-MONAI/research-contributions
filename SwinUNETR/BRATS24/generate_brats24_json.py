import json
import os
import glob
from sklearn.model_selection import KFold

def generate_brats24_json(base_dir, output_json_path, n_folds=5):
    '''
    Generates a JSON file for BraTS 2024 dataset folds.

    Args:
        base_dir (str): The root directory of the BraTS 2024 dataset.
                        This directory is expected to contain subdirectories for each patient.
        output_json_path (str): Path to save the generated JSON file.
        n_folds (int): Number of folds for cross-validation.
    '''
    patient_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    patient_dirs.sort()

    # This is a placeholder for how files might be named.
    # YOU WILL LIKELY NEED TO ADJUST THIS based on the actual filenames in the dataset.
    # Common BraTS modalities: t1ce, t1n, t2f, t2w. Segmentation file often ends with 'seg'.
    # Example: <patient_id>_t1ce.nii.gz, <patient_id>_seg.nii.gz

    # Assuming NIfTI files (.nii.gz)
    image_modalities_suffixes = ["_t1ce.nii.gz", "_t1n.nii.gz", "_t2flair.nii.gz", "_t2w.nii.gz"] # Common suffixes, verify with actual data
    label_suffix = "_seg.nii.gz" # Common suffix, verify

    all_files = []
    for patient_id in patient_dirs:
        patient_path = os.path.join(base_dir, patient_id)

        image_files = []
        # Try to find files with expected modality suffixes
        # This part needs to be robust and match actual file naming
        # For example, the dataset might have subfolders like 'images' and 'labels' or specific naming patterns.

        # Simplified example: assume files are directly in patient_path
        # And modalities are fixed as per image_modalities_suffixes
        # Users must adapt this glob pattern.

        # First, let's try to find the main files for each modality (t1ce, t1, t2, flair)
        # This part is highly dependent on the actual dataset structure and naming convention
        # For BRATS, typically you have something like:
        # BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1c.nii.gz
        # BraTS-GLI-00000-000/BraTS-GLI-00000-000-t1n.nii.gz
        # BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2f.nii.gz
        # BraTS-GLI-00000-000/BraTS-GLI-00000-000-t2w.nii.gz
        # BraTS-GLI-00000-000/BraTS-GLI-00000-000-seg.nii.gz

        # The kaggle dataset "nguyenthanhkhanh/brats2024-small-dataset" might have a different structure.
        # The user needs to run the notebook cell from Step 2 to find out the structure.
        # For now, I will assume a structure like:
        # <base_dir>/<patient_id>/<patient_id>_t1ce.nii.gz (and other modalities)
        # <base_dir>/<patient_id>/<patient_id>_seg.nii.gz

        # Let's assume the dataset has subdirectories for each patient
        # and inside each patient directory, there are files like:
        # patient_id_flair.nii.gz, patient_id_t1ce.nii.gz, patient_id_t1.nii.gz, patient_id_t2.nii.gz, patient_id_seg.nii.gz

        # Modality mapping based on typical BraTS datasets. This might need adjustment.
        # The order of modalities is important for the model's input channels.
        # Standard order: [FLAIR, T1c, T1, T2]
        modality_patterns = {
            "flair": f"{patient_id}/*_flair.nii.gz", # Or similar pattern
            "t1ce": f"{patient_id}/*_t1ce.nii.gz",   # Or _t1c.nii.gz
            "t1n": f"{patient_id}/*_t1n.nii.gz",    # Or _t1.nii.gz
            "t2w": f"{patient_id}/*_t2w.nii.gz"     # Or _t2.nii.gz
        }
        segmentation_pattern = f"{patient_id}/*_seg.nii.gz"

        img_paths_for_patient = []
        valid_patient = True

        # Try to find FLAIR, T1c, T1, T2 in order
        # This order is often expected by models trained on BraTS
        ordered_modalities = ["flair", "t1ce", "t1n", "t2w"]

        for mod_key in ordered_modalities:
            pattern = os.path.join(patient_path, modality_patterns[mod_key].split('/')[-1]) # Use only filename pattern
            found_files = glob.glob(pattern)
            if not found_files:
                print(f"Warning: Modality {mod_key} not found for patient {patient_id} with pattern {pattern}")
                valid_patient = False
                break
            img_paths_for_patient.append(found_files[0]) # Take the first match

        if not valid_patient:
            continue

        label_path_glob = os.path.join(patient_path, segmentation_pattern.split('/')[-1])
        found_labels = glob.glob(label_path_glob)
        if not found_labels:
            print(f"Warning: Segmentation not found for patient {patient_id} with pattern {label_path_glob}")
            continue

        label_path = found_labels[0]

        all_files.append({"image": img_paths_for_patient, "label": label_path, "id": patient_id})

    if not all_files:
        print(f"Error: No files found. Check base_dir '{base_dir}' and file naming patterns in the script.")
        return

    print(f"Found {len(all_files)} patient data entries.")

    # Create folds
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    dataset_folds = {"training": [], "validation": []} # For simplicity, not creating separate test set here.
                                                       # Original script uses fold for validation.

    # The original script structure for brats21_folds.json is a list of dictionaries,
    # where each dictionary can have a "fold" key.
    # Let's replicate that. The "training" key in the json seems to hold all data,
    # and the loader then picks the validation fold.

    output_data_list = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_files)):
        for i in val_idx:
            item = all_files[i].copy()
            item["fold"] = fold_idx
            output_data_list.append(item)
        for i in train_idx:
            # Items not in the current val_idx are implicitly training for this fold
            # The original data_utils.py handles this logic: if item["fold"] == current_fold then val, else train
            # So, we only need to mark the validation items with their fold number.
            # However, to be explicit and match the structure where 'training' contains all items:
            item = all_files[i].copy()
            if "fold" not in item: # ensure it's not already marked as a val item for another fold
                 # if an item is never in val_idx across all folds (if kf.split is exhaustive, this won't happen for all items)
                 # it will be part of training set for all folds.
                 # The logic in datafold_read is:
                 # if "fold" in d and d["fold"] == fold: val.append(d) else: tr.append(d)
                 # So, items NOT marked with the current fold become training data.
                 # To be safe, we can add all items and only mark validation items with a fold.
                 # Let's simplify: the JSON contains a list of all samples.
                 # Each sample that is part of a validation set for a specific fold will have a "fold": fold_idx entry.
                 pass # Already added if it was a validation item, or will be added if not

            # To ensure all items are in the list, let's just re-add all items but mark validation ones

    # Let's rebuild output_data_list to be clearer
    all_items_with_fold_info = []
    for i, file_entry in enumerate(all_files):
        is_validation_for_a_fold = False
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_files)):
            if i in val_idx:
                item_copy = file_entry.copy()
                item_copy["fold"] = fold_idx
                all_items_with_fold_info.append(item_copy)
                is_validation_for_a_fold = True
                break # Should only be validation for one fold if KFold is used this way
        if not is_validation_for_a_fold: # If it was never a validation sample (should not happen with KFold)
            all_items_with_fold_info.append(file_entry.copy())


    # The original `brats21_folds.json` has a top-level key (e.g., "Task01_BrainTumour_2021")
    # and under that, a "training" list which contains all samples.
    # Each sample in the "training" list might have a "fold" field.
    # Let's make it simpler: a single list under a "training" key.

    final_json_structure = {
        "brats2024_data": { # Or some other descriptive key
            "training": all_items_with_fold_info # This list will be used by datafold_read
        }
    }
    # The user will need to ensure datafold_read in data_utils_brats24.py correctly parses this.
    # Specifically, the `key` argument in `datafold_read` should match "brats2024_data".
    # And the list it iterates over should be `json_data[key]['training']`.

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(final_json_structure, f, indent=4)

    print(f"Successfully generated {output_json_path}")
    print("IMPORTANT: Review the generated JSON and the file paths within.")
    print("YOU MUST VERIFY AND POSSIBLY ADJUST THE FILE SEARCH PATTERNS (e.g., modality suffixes, directory structure) in this script.")

if __name__ == "__main__":
    # This is an example. User needs to replace with the actual path from kagglehub.
    # dataset_base_dir = "/path/to/downloaded/brats2024-small-dataset"
    # The notebook in step 2 will print this path.

    # For now, this script is a template. The user should run it after finding the dataset path.
    print("This script is a template to generate the brats24_folds.json.")
    print("1. Run the Jupyter notebook SwinUNETR_BraTS2024.ipynb to download the dataset and get its path.")
    print("2. Update the 'dataset_base_dir' variable in the __main__ block of this script with that path.")
    print("3. CRITICALLY REVIEW AND ADJUST the file/modality naming patterns inside the generate_brats24_json function.")
    print("4. Run this script: python SwinUNETR/BRATS24/generate_brats24_json.py")

    # Example usage (commented out by default):
    # kaggle_dataset_path = os.path.expanduser("~/.cache/kagglehub/datasets/nguyenthanhkhanh/brats2024-small-dataset/nguyenthanhkhanh_brats2024-small-dataset_1") # Example path
    # if os.path.exists(kaggle_dataset_path):
    #    generate_brats24_json(kaggle_dataset_path, "SwinUNETR/BRATS24/jsons/brats24_folds.json")
    # else:
    #    print(f"Example dataset path not found: {kaggle_dataset_path}. Please update and run manually.")
    pass
