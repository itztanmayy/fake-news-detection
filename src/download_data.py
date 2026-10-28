"""
Data Download Utility for Fake News Detection System.
Downloads:
1. ISOT Dataset (True.csv and Fake.csv)
2. LIAR Dataset (train.tsv, valid.tsv, test.tsv)
"""
import os
import urllib.request
from tqdm import tqdm

DATASET_URLS = {
    "data/raw/isot/True.csv": "https://raw.githubusercontent.com/jyojay/MONU_Project_4/main/Resources/True.csv",
    "data/raw/isot/Fake.csv": "https://raw.githubusercontent.com/jyojay/MONU_Project_4/main/Resources/Fake.csv",
    "data/raw/liar/train.tsv": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/train.tsv",
    "data/raw/liar/valid.tsv": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/valid.tsv",
    "data/raw/liar/test.tsv": "https://raw.githubusercontent.com/thiagorainmaker77/liar_dataset/master/test.tsv",
}

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url: str, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        print(f"[ALREADY EXISTS] {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")
        return
    print(f"Downloading {url} -> {output_path}...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(output_path, 'wb') as out_file:
        total_size = int(resp.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(output_path)) as pbar:
            while True:
                chunk = resp.read(1024 * 64)
                if not chunk:
                    break
                out_file.write(chunk)
                pbar.update(len(chunk))
    print(f"[DONE] Saved to {output_path} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")

def main():
    print("=" * 60)
    print("Fetching raw datasets for Fake News Detection System...")
    print("=" * 60)
    for path, url in DATASET_URLS.items():
        download_file(url, path)
    print("\nAll datasets downloaded and ready.")

if __name__ == "__main__":
    main()

