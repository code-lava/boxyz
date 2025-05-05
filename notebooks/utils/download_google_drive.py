import os
import json
import shutil
from pathlib import Path
import zipfile
from tqdm import tqdm
import requests
import re
from urllib.parse import urlencode


def extract_file_id(url):
    """
    Extract Google Drive file ID from various URL formats.

    Args:
        url: str: Google Drive URL

    Returns:
        str: File ID or None if not found
    """
    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
        r'/folders/([a-zA-Z0-9_-]+)'
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def download_gdrive_file(file_id, output_path):
    """
    Download file from Google Drive.

    Args:
        file_id: str: Google Drive file ID
        output_path: str: Path to save the file
    """
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()

    response = session.get(url, stream=True)

    if 'text/html' in response.headers.get('content-type', ''):
        try:
            confirm_match = re.search(r'name="confirm" value="([^"]+)"', response.text)
            uuid_match = re.search(r'name="uuid" value="([^"]+)"', response.text)

            if confirm_match and uuid_match:
                params = {
                    'id': file_id,
                    'export': 'download',
                    'confirm': confirm_match.group(1),
                    'uuid': uuid_match.group(1)
                }

                download_url = f"https://drive.usercontent.google.com/download?{urlencode(params)}"
                response = session.get(download_url, stream=True)
            else:
                token = None
                for key, value in response.cookies.items():
                    if key.startswith('download_warning'):
                        token = value
                        break

                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(url, params=params, stream=True)
        except Exception as e:
            print(f"Warning: Could not parse confirmation page: {e}")

    if response.status_code != 200:
        raise Exception(f"Failed to download file. Status code: {response.status_code}")

    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(block_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def extract_zip(zip_path, output_dir):
    """
    Extract zip file to output directory.

    Args:
        zip_path: str: Path to zip file
        output_dir: str: Path to output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        for file in tqdm(file_list, desc="Extracting"):
            zip_ref.extract(file, output_dir)


def download_and_extract(url, output_dir, cleanup=False):
    """
    Download Google Drive file and extract if it's a zip.

    Args:
        url: str: Google Drive URL
        output_dir: str: Path to output directory
        cleanup: bool: Whether to remove downloaded zip file after extraction
    """
    file_id = extract_file_id(url)
    if not file_id:
        raise ValueError('Could not extract file ID from URL')

    print(f"Extracted file ID: {file_id}")

    temp_dir = os.path.join(output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    download_path = os.path.join(temp_dir, f"{file_id}.download")

    print('Downloading from Google Drive...')
    download_gdrive_file(file_id, download_path)

    if zipfile.is_zipfile(download_path):
        print('File is a zip archive. Extracting...')
        extract_zip(download_path, output_dir)

        if cleanup:
            print('Cleaning up temporary files...')
            os.remove(download_path)
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
    else:
        final_path = os.path.join(output_dir, f"{file_id}.file")
        shutil.move(download_path, final_path)
        print(f"File saved to {final_path}")
        if not os.listdir(temp_dir):
            os.rmdir(temp_dir)

    print(f"Download complete. Output saved to {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download Google Drive files/folders and extract zips')
    parser.add_argument('--url', type=str, required=True, help='Google Drive URL')
    parser.add_argument('--output-dir', type=str, required=True, help='Path to output directory')
    parser.add_argument('--cleanup', action='store_true', help='Remove zip file after extraction')

    args = parser.parse_args()

    download_and_extract(args.url, args.output_dir, args.cleanup)