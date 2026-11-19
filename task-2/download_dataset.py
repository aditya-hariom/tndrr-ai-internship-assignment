"""
Download script for NIST AI 100-1 (AI Risk Management Framework 1.0)
Public domain document from the National Institute of Standards and Technology.
"""

import os
import requests

DATASET_URL = "https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf"
LOCAL_FILENAME = "NIST_AI_RMF_1.0.pdf"

def download_pdf(url: str = DATASET_URL, target_path: str = LOCAL_FILENAME) -> str:
    if os.path.exists(target_path):
        size = os.path.getsize(target_path)
        print(f"[Dataset] Document already exists at {target_path} ({size:,} bytes).")
        return target_path

    print(f"[Dataset] Downloading document from: {url}...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers, stream=True, timeout=60)
    response.raise_for_status()

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    size = os.path.getsize(target_path)
    print(f"[Dataset] Download complete: {target_path} ({size:,} bytes).")
    return target_path

if __name__ == "__main__":
    download_pdf()

