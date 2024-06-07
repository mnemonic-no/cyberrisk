import requests
import zipfile
import os
from pathlib import Path

NVD_FILEPATH = Path(__file__).parent.resolve() / "cve_corpus"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36",
}

def download_zipfile(year):
    url = f"https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-{year}.json.zip"
    try:
        response = requests.get(url, stream=True, headers=HEADERS)
        response.raise_for_status()
        filepath = NVD_FILEPATH / f"nvdcve-1.1-{year}.json.zip"
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("File downloaded and saved successfully.")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error occurred: {e}")
    except requests.exceptions.Timeout as e:
        print(f"Timeout occurred: {e}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_and_extract_cve_data(year):
    download_zipfile(year)
    zip_filepath = NVD_FILEPATH / f"nvdcve-1.1-{year}.json.zip"

    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(NVD_FILEPATH)
    os.remove(zip_filepath)
    print(f"Data for {year} extracted successfully.")

def main():
    NVD_FILEPATH.mkdir(parents=True, exist_ok=True)
    for year in range(2002, 2025):
        download_and_extract_cve_data(year)

if __name__ == "__main__":
    main()