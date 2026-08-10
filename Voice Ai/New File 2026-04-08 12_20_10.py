import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "--quiet"])

from huggingface_hub import login, hf_hub_download, list_repo_files
import shutil
import os

# Authenticate with HuggingFace
HF_TOKEN = "REVOKED_HF_TOKEN_SEE_ENV"
login(token=HF_TOKEN)
print("✅ HuggingFace logged in")

# Target dataset
REPO_ID = "bharatgenai/BhashaBench-Finance"
VOLUME_PATH = "/Volumes/arthasetu/bronze/uploads/"

print(f"\nListing files in {REPO_ID}...")
try:
    files = list_repo_files(REPO_ID, repo_type="dataset", token=HF_TOKEN)
    print(f"Found {len(files)} files in repo")
    
    # Filter for data files (parquet, csv, json)
    data_files = [f for f in files if any(f.endswith(ext) for ext in ['.parquet', '.csv', '.json', '.jsonl'])]
    print(f"Data files to download: {len(data_files)}")
    
    for file in data_files:
        print(f"\n  Downloading: {file}")
        
        # Download to local cache
        local_path = hf_hub_download(
            REPO_ID,
            filename=file,
            repo_type="dataset",
            token=HF_TOKEN
        )
        
        # Copy to Volume
        dest_filename = file.replace("/", "_")  # Flatten directory structure
        dest_path = os.path.join(VOLUME_PATH, dest_filename)
        shutil.copy(local_path, dest_path)
        
        print(f"  ✅ Saved to: {dest_path}")
    
    print(f"\n✅ Downloaded {len(data_files)} files to Volume")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you have access to the dataset on HuggingFace")
    print("2. Visit: https://huggingface.co/datasets/bharatgenai/BhashaBench-Finance")
    print("3. Click 'Request Access' if needed")
