import json

from typing import Dict, List, Optional, Any
from pathlib import Path
import datetime

import joblib
import pandas as pd
import torch
import torch.nn as nn
from cvss import CVSS3
from sklearn.preprocessing import MultiLabelBinarizer
from rich.progress import track
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertConfig,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CORPUS_FILEPATH = Path(".").resolve() / "cve_corpus"
MODELS_DIR = Path(__file__).parent.resolve() / "models"
MODEL_FILE =  MODELS_DIR / "cvss_distilbert.pth"
CVSS_METRIC_ORDER = ["AV", "AC", "PR", "UI", "S", "C", "I", "A"]
LABEL_ENCODERS = {
    "AV": {"N": 0, "L": 1, "A": 2, "P": 3},
    "AC": {"L": 0, "H": 1},
    "PR": {"N": 0, "L": 1, "H": 2},
    "UI": {"R": 0, "N": 1},
    "S": {"C": 0, "U": 1},
    "C": {"L": 0, "H": 1, "N": 2},
    "I": {"L": 0, "H": 1, "N": 2},
    "A": {"N": 0, "H": 1, "L": 2},
}


class CVSS_classifier:
    def __init__(self) -> None:
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        self.model = MultiOutputDistilBert(
            config, num_labels_list=[4, 2, 3, 2, 2, 3, 3, 3]
        )
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        if MODEL_FILE.is_file():
            state_dict = torch.load(MODEL_FILE, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            print(f"No CVSS model found in {MODELS_DIR}")

    def predict_cvss(self, text: str) -> Dict:
        self.model.eval()
        self.model.to(DEVICE)

        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )

        input_ids = tokens["input_ids"].to(DEVICE)
        attention_mask = tokens["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs["logits"]

        # Convert logits to actual labels using the label encoders
        cvss_vector = {}
        for idx, logit in enumerate(logits):
            predicted_idx = torch.argmax(logit, dim=-1).item()
            classifier_name = list(LABEL_ENCODERS.keys())[idx]
            label_decoder = {v: k for k, v in LABEL_ENCODERS[classifier_name].items()}
            cvss_vector[classifier_name] = label_decoder[predicted_idx]

        return cvss_vector


class MultiOutputDistilBert(DistilBertForSequenceClassification):
    def __init__(self, config, num_labels_list):
        super().__init__(config)
        self.pre_classifier = None
        self.classifier = None

        self.classifiers = nn.ModuleList(
            [nn.Linear(config.dim, num_labels) for num_labels in num_labels_list]
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]
        logits = [classifier(pooled_output) for classifier in self.classifiers]

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = sum(loss_fct(logit, label) for logit, label in zip(logits, labels))
            return {"loss": loss, "logits": logits} if return_dict else (loss, logits)

        return {"logits": logits} if return_dict else logits


def load_json_file(file_path: str):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("The file was not found at the specified path.")
        return None
    except json.JSONDecodeError:
        print("The file is not a valid JSON file.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def parse_cvss_vector(cvss_vector: str) -> Dict:
    mapping = {
        "AV": (
            "attack_vector",
            {"N": "NETWORK", "A": "ADJACENT", "L": "LOCAL", "P": "PHYSICAL"},
        ),
        "AC": ("attack_complexity", {"L": "LOW", "H": "HIGH"}),
        "PR": ("privileges_required", {"N": "NONE", "L": "LOW", "H": "HIGH"}),
        "UI": ("user_interaction", {"N": "NONE", "R": "REQUIRED"}),
        "S": ("scope", {"U": "UNCHANGED", "C": "CHANGED"}),
        "C": ("confidentiality_impact", {"N": "NONE", "L": "LOW", "H": "HIGH"}),
        "I": ("integrity_impact", {"N": "NONE", "L": "LOW", "H": "HIGH"}),
        "A": ("availability_impact", {"N": "NONE", "L": "LOW", "H": "HIGH"}),
    }

    components = cvss_vector.split("/")
    c = CVSS3(cvss_vector)
    extracted_data = {}

    for component in components[1:]:
        key, value = component.split(":")
        full_key, value_mapping = mapping[key]
        extracted_data[full_key] = value_mapping[value]

    extracted_data.update(
        {
            "cvssv3_version": components[0],
            "vector_string": cvss_vector,
            "base_score": c.scores()[0],
            "base_severity": c.severities()[0].upper(),
        }
    )

    return extracted_data


def extract_cvssv3_details(data: Dict) -> Dict:
    cvssv3 = data["impact"]["baseMetricV3"]["cvssV3"]

    extracted_data = {
        "cvssv3_version": cvssv3["version"],
        "vector_string": cvssv3["vectorString"],
        "attack_vector": cvssv3["attackVector"],
        "attack_complexity": cvssv3["attackComplexity"],
        "privileges_required": cvssv3["privilegesRequired"],
        "user_interaction": cvssv3["userInteraction"],
        "scope": cvssv3["scope"],
        "confidentiality_impact": cvssv3["confidentialityImpact"],
        "integrity_impact": cvssv3["integrityImpact"],
        "availability_impact": cvssv3["availabilityImpact"],
        "base_score": cvssv3["baseScore"],
        "base_severity": cvssv3["baseSeverity"],
    }

    return extracted_data


def count_each_tag_in_references(data) -> Dict[str, int]:
    references = data["cve"]["references"]["reference_data"]

    tag_counts: Dict[str, int] = {}
    for reference in references:
        for tag in reference["tags"]:
            ref_tag = "ref_" + tag.lower().replace(" ", "_").replace("/", "_")
            if ref_tag in tag_counts:
                tag_counts[ref_tag] += 1
            else:
                tag_counts[ref_tag] = 1

    return tag_counts


def contains_basemetricv3(cve_entry) -> bool:
    impact = cve_entry.get("impact", {})
    return "baseMetricV3" in impact


def extract_cvss_details(cve_entry: dict, cvss_clf, description: str) -> Dict[str, Any]:
    if contains_basemetricv3(cve_entry):
        return extract_cvssv3_details(cve_entry)
    else:
        cvss_metrics = cvss_clf.predict_cvss(description)
        cvss_vector = "CVSS:3.1/" + "/".join(
            f"{key}:{cvss_metrics[key]}" for key in CVSS_METRIC_ORDER
        )
        return parse_cvss_vector(cvss_vector)


def extract_cwe_list(cve_entry: dict) -> List[str]:
    problem_type_data = (
        cve_entry.get("cve", {}).get("problemtype", {}).get("problemtype_data", [])
    )
    if problem_type_data and problem_type_data[0].get("description", []):
        return [item["value"] for item in problem_type_data[0]["description"]]
    return []


def extract_data_from_cve(cve_entry: dict, cvss_clf) -> Optional[Dict[str, Any]]:
    cve_data = cve_entry.get("cve", {})
    cve_metadata = cve_data.get("CVE_data_meta", {})
    cve_id = cve_metadata.get("ID", None)
    description_data = cve_data.get("description", {}).get("description_data", [])
    description = description_data[0]["value"] if description_data else ""

    cvss = extract_cvss_details(cve_entry, cvss_clf, description)
    cwe_list = extract_cwe_list(cve_entry)

    tag_counts = count_each_tag_in_references(cve_entry)

    return {
        "cve_id": cve_id,
        "description": description,
        "published_date": cve_entry.get("publishedDate", ""),
        "last_modified_date": cve_entry.get("lastModifiedDate", ""),
        "cwe_list": cwe_list,
        **tag_counts,
        **cvss,
    }


def is_reject_cve(cve_json) -> bool:
    reject_phrases = ["** REJECT **", "DO NOT USE THIS CANDIDATE NUMBER"]
    description = (
        cve_json.get("cve", {})
        .get("description", {})
        .get("description_data", [{}])[0]
        .get("value", "")
    )
    if any(phrase in description for phrase in reject_phrases):
        return True
    else:
        return False


def process_cve_entries(nvd_filepath) -> List:
    cvss_clf = CVSS_classifier()
    cve_data = []
    for file_path in sorted(nvd_filepath.iterdir()):
        print("Processing: ", str(file_path))
        corpus = load_json_file(file_path)
        for cve_entry in track(corpus["CVE_Items"]):
            if is_reject_cve(cve_entry):
                continue
            extracted_data = extract_data_from_cve(cve_entry, cvss_clf)
            if extracted_data:
                cve_data.append(extracted_data)
    return cve_data


def preprocess_cwe_data(cve_data: List[dict]) -> pd.DataFrame:
    """Extract and transform CWE data using MultiLabelBinarizer."""
    mlb = MultiLabelBinarizer()
    cwe_lists = [entry.pop("cwe_list", []) for entry in cve_data]
    transformed_data = mlb.fit_transform(cwe_lists)
    joblib.dump(mlb, MODELS_DIR / "cwe_binarizer.joblib")
    return pd.DataFrame(transformed_data, columns=mlb.classes_)


def create_dataframe(cve_data: List[dict]) -> pd.DataFrame:
    experiment_string = get_experiment_string()
    file_path = Path(".").resolve() / "data" / f"preprocessed_{experiment_string}.csv"
    cwe_df = preprocess_cwe_data(cve_data)
    existing_data = pd.DataFrame(cve_data)
    final_df = pd.concat([existing_data, cwe_df], axis=1)

    final_df.to_csv(file_path, index=False)


def get_experiment_string(args=None) -> str:
    experiment_string = str(datetime.datetime.now()).replace(" ", "_")
    if args:
        for k, v in args.__dict__.items():
            experiment_string += f"{k}-{str(v)}"
    return experiment_string


def write_to_file(cve_entries):
    experiment_string = get_experiment_string()
    file_path = Path(".").resolve() / "data" / f"cve_entries_{experiment_string}.json"
    with open(file_path, "w") as file:
        for dictionary in cve_entries:
            file.write(json.dumps(dictionary) + "\n")


def read_dicts_from_file(file_path):
    with open(file_path, "r") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    cve_data = process_cve_entries(CORPUS_FILEPATH)
    write_to_file(cve_data)
    create_dataframe(cve_data)


if __name__ == "__main__":
    main()
