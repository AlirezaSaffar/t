import os
import numpy as np
from collections import defaultdict

def read_numeric_file(path):
    """
    فرض می‌کنیم فایل شامل اعداد float جدا شده با space است.
    خروجی: numpy array 1D
    """
    try:
        data = np.loadtxt(path)
        return data
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None

def profile_modality(folder_path, modality_name, sample_limit=None):
    files = sorted([
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ])
    if sample_limit:
        files = files[:sample_limit]

    modality_report = {
        "modality": modality_name,
        "num_files": len(files),
        "file_shapes": [],
        "min_vals": [],
        "max_vals": [],
        "mean_vals": [],
        "std_vals": [],
        "nan_counts": [],
        "error_files": []
    }

    for f in files:
        data = read_numeric_file(f)
        if data is None:
            modality_report["error_files"].append(f)
            continue
        modality_report["file_shapes"].append(data.shape)
        modality_report["min_vals"].append(np.min(data))
        modality_report["max_vals"].append(np.max(data))
        modality_report["mean_vals"].append(np.mean(data))
        modality_report["std_vals"].append(np.std(data))
        modality_report["nan_counts"].append(np.sum(np.isnan(data)))

    modality_report["overall_min"] = float(np.min(modality_report["min_vals"])) if modality_report["min_vals"] else None
    modality_report["overall_max"] = float(np.max(modality_report["max_vals"])) if modality_report["max_vals"] else None
    modality_report["overall_mean"] = float(np.mean(modality_report["mean_vals"])) if modality_report["mean_vals"] else None
    modality_report["overall_std"] = float(np.mean(modality_report["std_vals"])) if modality_report["std_vals"] else None

    return modality_report

def profile_dataset_numeric(root_dir, sample_limit=None):
    """
    root_dir باید مسیر پوشه _2021-08-06-10-59-33 باشد
    هر زیرپوشه modality مختلف است: rgb, nir, thr, lidar
    """
    modalities = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    report = {}
    for mod in modalities:
        folder_path = os.path.join(root_dir, mod)
        report[mod] = profile_modality(folder_path, mod, sample_limit)
    return report

def print_report_numeric(report):
    print("\n================ NUMERIC DATASET PROFILE ================\n")
    for mod, r in report.items():
        print(f"📌 Modality: {mod}")
        print(f"  - Number of files: {r['num_files']}")
        print(f"  - File shapes (sample 5): {r['file_shapes'][:5]}")
        print(f"  - Overall Min: {r['overall_min']}")
        print(f"  - Overall Max: {r['overall_max']}")
        print(f"  - Overall Mean: {r['overall_mean']:.4f}")
        print(f"  - Overall Std : {r['overall_std']:.4f}")
        print(f"  - Total NaN values: {sum(r['nan_counts'])}")
        if r['error_files']:
            print(f"  - Files with errors: {r['error_files']}")
        print("--------------------------------------------------------")
    print("\n========================================================\n")


dataset_path = "_2021-08-06-10-59-33"
report = profile_dataset_numeric(dataset_path, sample_limit=1000) 
print_report_numeric(report)

