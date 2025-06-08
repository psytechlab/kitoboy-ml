# https://python-bloggers.com/2022/08/hyperparameter-tuning-a-transformer-with-optuna/

import hydra
from omegaconf import OmegaConf
import optuna
import pandas as pd
import json
import logging
import os
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import sys

WORKING_DIR = os.getcwd()
DEFAULT_CONFIG = "training_config.yaml"
sys.path.append(os.getcwd())
from src.utils.clearml_utils import get_clearml_dataset_by_id
from src.utils.utils import save_text, load_json
from src.scripts.run_training import prepare_data, prepare_dataset, compute_metrics



@hydra.main(config_path=f"{WORKING_DIR}/conf", config_name=DEFAULT_CONFIG)
def main(cfg: OmegaConf):
    
    report_to_arg = "none"
    if cfg.dataset.download_from_clearml:
        logging.info(f"Download dataset id={cfg.dataset.dataset_id} from clearml")
        dataset_path = get_clearml_dataset_by_id(cfg.dataset.dataset_id)
        cfg.dataset.base_path = str(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_path)
    logging.info("Preparing data")
    train_dataset, test_dataset, eval_dataset, label2id = prepare_data(cfg, tokenizer)
    if eval_dataset is None:
        evaluation_strategy = "no"
        load_best_model_at_end = None
    else:
        evaluation_strategy = "epoch"
        load_best_model_at_end = True
        
    def objective(trial: optuna.Trial):     
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.model_path, num_labels=len(label2id), label2id=label2id
        )     
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=cfg.model.batch_size_train,
            per_device_eval_batch_size=cfg.model.batch_size_eval,
            weight_decay=trial.suggest_float('weight_decay', WD_MIN, WD_CEIL),  
            learning_rate=trial.suggest_float('learning_rate', low=LR_MIN, high=LR_CEIL),
            max_grad_norm=trial.suggest_categorical('max_grad_norm', MAX_GRAD_NORM),
            warmup_ratio=trial.suggest_float('warmup_ratio', low=WARMUP_MIN, high=WARMUP_CEIL),
            lr_scheduler_type=trial.suggest_categorical("lr_scheduler", LRS),
            label_smoothing_factor=trial.suggest_float('label_smoothing_factor', low=LABEL_SMOOTHING_MIN, high=LABEL_SMOOTHING_CEIL),
            evaluation_strategy="no",  # Валидация после каждой эпохи (можно сделать после конкретного кол-ва шагов)
            logging_strategy="no",  # Логирование после каждой эпохи
            save_strategy="no",  # Сохранение после каждой эпохи
            report_to=report_to_arg,
            seed=cfg.seed,
            fp16=False
            )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate(eval_dataset)
        print(metrics['eval_f1_macro'])
        return metrics['eval_f1_macro']

        
    LR_MIN = 1e-5
    LR_CEIL = 1e-4
    WD_MIN = 0.001
    WD_CEIL = 0.02
    MAX_GRAD_NORM = [1,3,5]
    LRS = ["linear","cosine", "polynomial"]
    WARMUP_MIN = 0.0
    WARMUP_CEIL = 0.1
    LABEL_SMOOTHING_MIN = 0.0
    LABEL_SMOOTHING_CEIL = 1.0

    NUM_TRIALS = 1
    study = optuna.create_study(study_name='hp-search-antisui_model', direction='maximize') 
    study.optimize(func=objective, n_trials=NUM_TRIALS, n_jobs=1)
    print(study.best_params)
    df = study.trials_dataframe()
    df.to_csv("/workspace/kitoboy-ml/data/presui_optuna.csv")
    with open("/workspace/kitoboy-ml/data/presui_bert_params.json", 'w') as f:
        json.dump(study.best_params, f)
        
if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    main()
