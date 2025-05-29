"""The main script for training the model"""
import hydra
import torch
import pandas as pd
import logging
import clearml
import os
import numpy as np
import shutil
from typing import List, Tuple
from torch.utils.data import Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import transformers
from omegaconf import OmegaConf
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from textfab.fabric import Fabric

from collections import Counter



import sys

WORKING_DIR = os.getcwd()
DEFAULT_CONFIG = "training_config.yaml"
sys.path.append(WORKING_DIR)
from src.utils.clearml_utils import get_clearml_dataset_by_id
from src.utils.custom_textfab_units import replace_ru_e
from src.utils.utils import save_text, load_json



logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


CUSTOM_PREPROCESSORS = {
    "replace_ru_e": replace_ru_e
}


class TextsLabelsDataset(Dataset):
    """The class for storing dataset for training."""

    def __init__(self, texts: List[str], labels: List[int]) -> None:
        """Init.

        Args:
            texts (List[str]): The list of texts.
            labels (List[int]): The list of labels.
        """
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            int: The length.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> transformers.BatchEncoding:
        """Getitem implementation.

        Args:
            idx (int): The index to be get.

        Returns:
            ransformers.BatchEncoding: A complete instance 
            to be fed into trainer.
        """
        item = {k: v[idx] for k, v in self.texts.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


class TextsLabelsDataCollator:
    """The custom collator for TextLabelsDataset."""

    def __init__(self, tokenizer: transformers.AutoTokenizer) -> None:
        """Init.

        Args:
            tokenizer (transformers.AutoTokenizer): The tokenizer to be used.
        """
        self.tokenizer = tokenizer

    def __call__(self, batch: List[Tuple[str, int]]) -> torch.Tensor:
        """Form the batch from the dataset.

        Args:
            batch (List[Tuple[str, int]]): The raw batch data.

        Returns:
            torch.Tensor: Batch in tensor representation.
        """
        texts, labels = [sample[0] for sample in batch], [sample[1] for sample in batch]
        batch = self.tokenizer.batch_encode_plus(
            *texts,
            padding="longest",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        if isinstance(labels[0], str) is not True:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def compute_metrics(eval_pred) -> dict[str, float]:
    """Compute f1 metrics.

    Args:
        eval_pred (_): The predictions from the Trainer.

    Returns:
        dict[str, float]: The dict with differently averaged F1.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    metrics = {
        "f1_micro": f1_score(labels, preds, average="micro"),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }
    return metrics


def additional_metrics(eval_pred, cfg: OmegaConf, label2id: dict[str, int], file_name: str ="test") -> None:
    """Compute the classification report.

    Args:
        eval_pred (_type_): The predictions from the Trainer.
        cfg (OmegaConf): The Hydra config file.
        label2id (dict[str, int]): The mapping from labels to id.
    """
    output_dir = Path(cfg.output_dir)

    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    preds = np.argmax(logits, axis=-1)
    

    cr = classification_report(labels, preds, target_names=sorted(list(label2id)), 
                               zero_division=0, labels=sorted(list(label2id.values())))
    
    save_text(cr, output_dir / f"cr_{file_name}.txt")
    
    report = classification_report(labels, preds, output_dict=True, zero_division=0, 
                                   target_names=sorted(list(label2id)), 
                                   labels=sorted(list(label2id.values())))
    
    df = pd.DataFrame(report).transpose().sort_values('f1-score', ascending=False)
    df.to_csv(output_dir / f"cr_{file_name}.csv")
    
    cf_matrix = confusion_matrix(labels, preds)
    cf_matrix_norm = (
            cf_matrix.astype("float") / cf_matrix.sum(axis=1)[:, np.newaxis]
        )
    
    
    target_names = list(label2id)

    fig, ax = plt.subplots(figsize=(25, 17))
    sns.heatmap(
            cf_matrix_norm,
            annot=True,
            fmt=".2f",
            xticklabels=target_names,
            yticklabels=target_names,
    )
    plt.ylabel("Gold")
    plt.xlabel("Predicted")
    fig.savefig(
        output_dir / f"cf_{file_name}.png",
        bbox_inches="tight",
        facecolor="white",
        transparent=True,
    )

    return df


def prepare_dataset(
    df: pd.DataFrame, cfg: OmegaConf, tokenizer: transformers.AutoTokenizer, preprocessor: Fabric
) -> TextsLabelsDataset:
    """Prepare the dataset from training.

    Args:
        df (pd.DataFrame): The source dataset in table.
        cfg (OmegaConf): The hydra config.
        tokenizer (transformers.AutoTokenizer): Tokenizer to be used in training.

    Returns:
        TextsLabelsDataset: The dataset object.
    """
    df[cfg.dataset.text_col] = df[cfg.dataset.text_col].astype(str).apply(lambda x: preprocessor([x])[0])
    texts = df[cfg.dataset.text_col].to_list()
    tokenized_texts = tokenizer(
        texts,
        return_tensors="pt",
        truncation=cfg.truncation,
        max_length=cfg.max_length,
        padding=cfg.padding,
    )
    labels = df[cfg.dataset.label_col].to_list()
    return TextsLabelsDataset(tokenized_texts, labels)


def prepare_data(
    cfg: OmegaConf, tokenizer: transformers.AutoTokenizer
) -> tuple[TextsLabelsDataset, TextsLabelsDataset, TextsLabelsDataset]:
    """Prepare all the data needed for training.

    Args:
        cfg (OmegaConf): Hydra config
        tokenizer (_type_): The tokenizer to be used in training.

    Returns:
        tuple[TextsLabelsDataset, TextsLabelsDataset, TextsLabelsDataset]: The three datasets ready to be used in Trainer.
    """
    base_path = Path(cfg.dataset.base_path)
    
    fab_preprocessor = Fabric(
        [
            *cfg.preprocessing.primitives, 
            *[CUSTOM_PREPROCESSORS[p]() for p in cfg.preprocessing.custom]
        ]
    )
    print('fab_preprocessor: ', fab_preprocessor)
 

    label2id = load_json(base_path / cfg.dataset.label2id)
    label2id = {k: int(v) for k, v in label2id.items()}
    label2id = [(k, v) for k, v in label2id.items()]
    label2id.sort(key=lambda x: x[1])
    label2id = dict(label2id)

    df_train = pd.read_csv(base_path / cfg.dataset.train_file, sep=cfg.dataset.sep)
    df_test = pd.read_csv(base_path / cfg.dataset.test_file, sep=cfg.dataset.sep)

    if cfg.test_run:
        df_train = df_train.sample(16)

    df_train[cfg.dataset.label_col] = df_train[cfg.dataset.label_col].map(label2id)
    df_test[cfg.dataset.label_col] = df_test[cfg.dataset.label_col].map(label2id)
    train_dataset = prepare_dataset(df_train, cfg, tokenizer, fab_preprocessor)
    test_dataset = prepare_dataset(df_test, cfg, tokenizer, fab_preprocessor)
    if cfg.dataset.eval_file is not None:
        df_eval = pd.read_csv(base_path / cfg.dataset.eval_file, sep=cfg.dataset.sep)
        df_eval[cfg.dataset.label_col] = df_eval[cfg.dataset.label_col].map(label2id)
        eval_dataset = prepare_dataset(df_eval, cfg, tokenizer, fab_preprocessor)
    elif cfg.dataset.use_test_as_eval:
        eval_dataset = test_dataset
    else:
        eval_dataset = None

    return train_dataset, test_dataset, eval_dataset, label2id


@hydra.main(config_path=f"{WORKING_DIR}/conf", config_name=DEFAULT_CONFIG)
def main(cfg: OmegaConf):
    """Implement main.

    Args:
        cfg (OmegaConf): Hydra config
    """
    print("Config: ", cfg)
    report_to_arg = "none"
    
    if cfg.dataset.download_from_clearml:
        logging.info(f"Download dataset id={cfg.dataset.dataset_id} from clearml")
        dataset_path = get_clearml_dataset_by_id(cfg.dataset.dataset_id)
        cfg.dataset.base_path = str(dataset_path)
        
    if cfg.log_into_clearml:
        report_to_arg = "clearml"
        clearml_task = clearml.Task.init(
            project_name=cfg.clearml.project,
            task_name=cfg.clearml.task_name,
            task_type=cfg.clearml.task_type,
            output_uri=False,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path)
    logging.info("Preparing data")
    train_dataset, test_dataset, eval_dataset, label2id = prepare_data(cfg, tokenizer)
    if eval_dataset is None:
        evaluation_strategy = "no"
        load_best_model_at_end = None
    else:
        evaluation_strategy = "epoch"
        load_best_model_at_end = True

    logging.info("Loading model")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model.model_path, num_labels=len(label2id), label2id=label2id
    )
    if cfg.model.freeze_encoder:
        for param in model.bert.parameters():
            param.requires_grad = False


    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=cfg.model.epoch,
        per_device_train_batch_size=cfg.model.batch_size_train,
        per_device_eval_batch_size=cfg.model.batch_size_eval,
        weight_decay=cfg.model.weight_decay,
        logging_dir=cfg.logging_dir,
        load_best_model_at_end=load_best_model_at_end,  # Загружать ли лучшую модель после обучения
        learning_rate=cfg.model.learning_rate,
        evaluation_strategy=evaluation_strategy,  # Валидация после каждой эпохи (можно сделать после конкретного кол-ва шагов)
        logging_strategy="steps",  # Логирование после каждой эпохи
        logging_steps=50,
        save_strategy="epoch",  # Сохранение после каждой эпохи
        save_total_limit=cfg.save_total_limit,
        report_to=report_to_arg,
        seed=cfg.seed,
        metric_for_best_model=cfg.metric_for_best_model
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    metrics = trainer.evaluate(test_dataset)
    test_metrics = pd.DataFrame([['test', metrics['eval_f1_micro'], metrics['eval_f1_macro'], 
                                  metrics['eval_f1_weighted'], metrics['eval_loss'], len(test_dataset)]],
             columns = ['set', 'f1_micro', 'f1_macro', 'f1_weighted', 'loss', 'n'])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    preds = trainer.predict(test_dataset)
    cr_test_df = additional_metrics(preds, cfg, label2id)
    
    eval_metrics = trainer.evaluate(eval_dataset)
    eval_metrics = pd.DataFrame([['val', eval_metrics['eval_f1_micro'], eval_metrics['eval_f1_macro'], 
                                  eval_metrics['eval_f1_weighted'], eval_metrics['eval_loss'], len(eval_dataset)]],
             columns = ['set', 'f1_micro', 'f1_macro', 'f1_weighted', 'loss', 'n'])
    
    overall_metrics = pd.concat([test_metrics, eval_metrics])
    overall_metrics.to_csv(Path(cfg.output_dir) / "overall_metrics.csv")
    
    
    cr_val_df = additional_metrics(trainer.predict(eval_dataset), cfg, label2id, file_name = 'val')
    
    excel_path = Path(cfg.output_dir) / "report.xls"
    with pd.ExcelWriter(excel_path) as w:  # pylint: disable=abstract-class-instantiated
        overall_metrics.to_excel(w, sheet_name="overall", index=False)
        cr_test_df.to_excel(w, sheet_name="cr_test", index=False)
        cr_val_df.to_excel(w, sheet_name="cr_val", index=False)


    model_save_path = Path(f"{cfg.output_dir}/best_model")
    print('model_save_path: ', model_save_path)
    model_save_path.mkdir(exist_ok=True, parents=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    # remove all checkpoint except best_model
    if cfg.remove_checkpoints:
        for d in os.listdir(cfg.output_dir):
            if d != "best_model":
                # Try to remove the tree; if it fails, throw an error using try...except.
                try:
                    shutil.rmtree(os.path.join(cfg.output_dir, d))
                except OSError as e:
                    print("Error: %s - %s." % (e.filename, e.strerror))
    if cfg.log_into_clearml:
        out_model = clearml.OutputModel(
            task=clearml_task, name=f"{cfg.clearml.task_name}-model", framework="Torch"
        )
        out_model.set_upload_destination(cfg.clearml.model_upload_dest)
        out_model.update_weights(weights_filename=model_save_path, auto_delete_file=False)



if __name__ == "__main__":

    main()
