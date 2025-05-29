"""The script allows to download published models and datasets and makes validation.

Usage exeamples:

    Run script to validate andisui model:
        python src/scripts/validate_published_models.py --dataset_hf_repo_id=psytechlab/antisuisidal_dataset --model_hf_repo_id=psytechlab/antisuisidal_model
        
     Run script to validate presui model:        
    python src/scripts/validate_published_models.py --dataset_hf_repo_id=psytechlab/presuisidal_dataset --model_hf_repo_id=psytechlab/presuisidal_model
"""

import argparse
import os
import sys

from time import time
import pandas as pd
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, BertForSequenceClassification


WORKING_DIR = os.getcwd()
sys.path.append(WORKING_DIR)
from src.utils.utils import load_json, save_to_json
from src.utils.text_features import apply_model, preprocess


def main(args):
    """
    The main function to help:
    1. Download dataset by given dataset_hf_repo_id;
    2. Download model by given model_hf_repo_id;
    3. Validate the model on the test and show f1-macro.
    """
    # Parse args
    dataset_hf_repo_id = args.dataset_hf_repo_id
    model_hf_repo_id = args.model_hf_repo_id 
    device = args.device 
    
    # Loading training dataset and label2id.json
    print(f"Loading {dataset_hf_repo_id}...")
    dataset = datasets.load_dataset(dataset_hf_repo_id)
    print(f"{dataset_hf_repo_id}: ", dataset)

    
    print(f"Loading {dataset_hf_repo_id} label2id.json ...")
    label2id_path = hf_hub_download(repo_id=dataset_hf_repo_id, 
                filename="label2id.json", 
                repo_type="dataset")
    label2id = load_json(label2id_path)
    print("Classes: ", label2id)
          
    # Loading model
    print(f"Loading {model_hf_repo_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_hf_repo_id)
    model = BertForSequenceClassification.from_pretrained(model_hf_repo_id)
    model = model.to(device)
    model.eval()
 
    
    # Validation model on test set
    texts = dataset["test"]["text"]
    true_label_names = dataset["test"]["label"]
    true_label_ids = list(map(lambda x: label2id[x], true_label_names))
    
    predictied_label_ids, predicted_scores = apply_model(
        tokenizer=tokenizer, model=model, text_list=preprocess(texts)
    )
    

    cr = classification_report(true_label_ids, predictied_label_ids, output_dict=True, zero_division=0, 
                                   target_names=sorted(list(label2id)), 
                                   labels=sorted(list(label2id.values())))

    f1_macro = cr["macro avg"]["f1-score"]
    print("F1 macro: ", f1_macro)
    return f1_macro
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_hf_repo_id",
        type=str,
        required=True,
        help="The training dataset repo id in HuggingFace"
    )
    parser.add_argument(
        "--model_hf_repo_id",
        type=str,
        required=True,
        help="The model repo id in HuggingFace"
    )
    parser.add_argument(
        "--device",
        choices=["cuda"],
        default="cuda",
        help="Only cuda available"
    )

    args = parser.parse_args()
    main(args)
