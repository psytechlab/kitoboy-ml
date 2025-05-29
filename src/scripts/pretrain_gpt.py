import hydra
import clearml
from pathlib import Path
import tokenizers
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

import logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

import sys
import os
sys.path.append(os.getcwd())
from src.utils.clearml_utils import get_clearml_dataset_by_id, save_model_clearml


def preprocess_function(examples: Dataset, tokenizer: AutoTokenizer, col_name: str, max_len: int):
    """Concatenate all texts from Datasets.

    Args:
        examples (Dataset): The dataset with texts.
        tokenizer (AutoTokenizer): The tokenizer to be used.
        col_name (str): Name of column with text in `examples` dataset.
        max_len (int): Max length of the text.

    Returns:
        BatchEncoding: Tokenized text
    """
    return tokenizer([" ".join(x) for x in examples[col_name]], max_length=max_len, truncation=True, padding=True)

def group_texts(examples, block_size):
    """Group texts into chunkts to be fed into model.

    Args:
        examples (Dataset): The dataset with texts
        block_size (int): The maximum block size.

    Returns:
        dict: The dataset with grouped text.
    """
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

@hydra.main(config_path=f"{os.getcwd()}/conf", config_name="pretrain_config_gpt.yaml")
def main(cfg):
    """Run the pretraining of the GPT

    Args:
        cfg (OmegaConf): The Hydra config the parameters. See `config` directory.
    """
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
    model = AutoModelForCausalLM.from_pretrained(cfg.model.model_path)
    
    dataset = load_dataset("text", data_files={"train": [f"{cfg.dataset.base_path}/{cfg.dataset.data_file}"]})

    tokenized_dataset = dataset.map(
                    lambda x: preprocess_function(x, tokenizer, cfg.data_col_name, cfg.max_len),
                    batched=True,
                    num_proc=cfg.num_proc,
                    remove_columns=dataset["train"].column_names,
                )
    lm_dataset = tokenized_dataset.map(lambda x: group_texts(x, cfg.block_size), batched=True, num_proc=cfg.num_proc)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        overwrite_output_dir=True,
        learning_rate=cfg.model.learning_rate,
        num_train_epochs=cfg.model.epoch,
        weight_decay=cfg.model.weight_decay,
        per_device_train_batch_size=cfg.model.batch_size_train,
        gradient_accumulation_steps=cfg.model.gradient_accumulation_steps,
        logging_strategy='steps',
        logging_steps=100,
        save_strategy='epoch',
        save_total_limit=1,
        push_to_hub=False,
        report_to=report_to_arg
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        #eval_dataset=lm_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train()

    model_save_path = Path(f"{cfg.output_dir}/best_model")
    model_save_path.mkdir(exist_ok=True, parents=True)
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    logging.info(f"Saving the model")
    out_model = clearml.OutputModel(task=clearml_task, name=f"{cfg.clearml.task_name}-model", framework="Torch")
    out_model.set_upload_destination("")
    out_model.update_weights(weights_filename=model_save_path)

if __name__ == "__main__":
    main()

