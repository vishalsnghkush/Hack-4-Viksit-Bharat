
import os
import sys
from pathlib import Path
import urllib.request
import zipfile
import shutil

# Dataset Config
DATASETS = {
    "UrbanNav": {
        "url": "https://github.com/IPNL-POLYU/UrbanNavDataset", # Placeholder - manual download typically needed for large files
        "description": "GPS-Denied Urban Canyon Data (Hong Kong)",
        "path": "data/datasets/UrbanNav"
    },
    "Cityscapes": {
        "url": "https://www.cityscapes-dataset.com/",
        "description": "Semantic Segmentation for Urban Scenes",
        "path": "data/datasets/Cityscapes",
        "instructions": "Requires login. We will use HuggingFace pre-trained models instead."
    },
    "Alpamayo": {
        "url": "https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles",
        "description": "NVIDIA Physical AI AV Dataset (Reasoning)",
        "path": "data/datasets/Alpamayo",
        "instructions": "Huge dataset (>1TB). Use HuggingFace 'datasets' library to stream samples."
    }
}

def setup_directories(project_root: Path):
    """Create directory structure for datasets."""
    print(f"Setting up directories in {project_root}")
    
    for name, info in DATASETS.items():
        dataset_path = project_root / info["path"]
        if not dataset_path.exists():
            print(f"Creating directory for {name}: {dataset_path}")
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Create README
            with open(dataset_path / "README.txt", "w") as f:
                f.write(f"Dataset: {name}\n")
                f.write(f"Description: {info['description']}\n")
                f.write(f"URL: {info['url']}\n")
                if "instructions" in info:
                    f.write(f"Note: {info['instructions']}\n")
        else:
            print(f"Directory exists: {dataset_path}")

def main():
    project_root = Path(__file__).parent.parent
    setup_directories(project_root)
    
    print("\n" + "="*50)
    print("DATASET SETUP COMPLETE")
    print("="*50)
    print("Note: Most autonomous vehicle datasets (Cityscapes, Alpamayo)")
    print("are too large to auto-download (>100GB).")
    print("\nWe will use:")
    print("1. PRE-TRAINED MODELS (SegFormer) which 'contain' the Cityscapes knowledge.")
    print("2. STREAMING from HuggingFace for Alpamayo samples if needed.")
    print("3. UrbanNav samples for Visual Odometry calibration.")
    print("="*50)

if __name__ == "__main__":
    main()
