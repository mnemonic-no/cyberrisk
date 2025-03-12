from pathlib import Path
import json
from typing import List, Dict, Optional, Any

import pandas as pd

from config import NVD_FILEPATH, TRAINING_DIR




def load_json_data(json_filepath: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(json_filepath, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return None


def extract_training_set(data_dir: Path, nvd_filepath: Path) -> None:
    CVSS_DIR = data_dir / "cvss"
    CWE_DIR = data_dir / "cwe"

    for directory in [CVSS_DIR, CWE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    cvss_entries: List[Dict[str, str]] = []
    cwe_entries: List[Dict[str, str]] = []
    unique_cwes: set[str] = set()

    for json_file in Path(nvd_filepath).glob("*.json"):
        cve_data = load_json_data(json_file)
        if cve_data is None:
            continue

        for entry in cve_data["CVE_Items"]:
            if "baseMetricV3" not in entry["impact"]:
                continue

            description = entry["cve"]["description"]["description_data"][0]["value"]
            cve_id = entry["cve"]["CVE_data_meta"]["ID"]

            # CVSS data
            vector_string = entry["impact"]["baseMetricV3"]["cvssV3"]["vectorString"]
            metrics = vector_string.split("/")[1:]
            metric_dict: Dict[str, str] = {}
            for metric in metrics:
                metric_name, metric_value = metric.split(":")
                metric_dict[metric_name] = metric_value

            cvss_file_path = CVSS_DIR / f"{cve_id}_cvss.txt"
            cvss_file_path.write_text(description)

            cvss_entry = {
                "filename": cvss_file_path.name,
                "cve_id": cve_id,
                **metric_dict,
            }
            cvss_entries.append(cvss_entry)

            # CWE data
            cwes: List[str] = []
            if "problemtype" in entry["cve"]:
                for problem in entry["cve"]["problemtype"]["problemtype_data"]:
                    for desc in problem["description"]:
                        if desc["lang"] == "en" and desc["value"].startswith("CWE-"):
                            cwes.append(desc["value"])
                            unique_cwes.add(desc["value"])

            if cwes:  # Only create CWE entry if there are CWEs
                cwe_file_path = CWE_DIR / f"{cve_id}_cwe.txt"
                cwe_file_path.write_text(description)

                cwe_entry = {
                    "filename": cwe_file_path.name,
                    "cve_id": cve_id,
                    "cwes": json.dumps(cwes),
                }
                cwe_entries.append(cwe_entry)

    cvss_df = pd.DataFrame(cvss_entries)
    cvss_df.to_csv(CVSS_DIR / "cvss_metadata.csv", index=False)

    cwe_df = pd.DataFrame(cwe_entries)
    cwe_df.to_csv(CWE_DIR / "cwe_metadata.csv", index=False)

    print(f"\nDataset Statistics:")
    print(f"Total unique CWEs found: {len(unique_cwes)}")
    print(f"Total entries with CWEs: {len(cwe_entries)}")
    print(f"Total CVSS entries: {len(cvss_entries)}")

if __name__ == "__main__":
    extract_training_set(data_dir=TRAINING_DIR, nvd_filepath=NVD_FILEPATH)