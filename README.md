# kitoboy-ml



# Scripts

Here the description of all runnable scripts is presented. These scripts is places in `./src/scripts` directory.

## label_studio_cli.py

The script is designed to interact with Label Studio (LS for short). It allows you to download or deploy annotation data for a classification tasks which labeling interface is structured as text to be annotated, label set and comment field. 

You have to provide the settings file in which the host with LS is defined as well as a dictionary with ports as keys and corresponding access tokens as values. By default it placed in `reference/settings.yaml`. The intuition behind that is the setting with multiple community edition LS deployed on one server. This is because this version LS doesn't allow to do annotation in parallel in one project. So you assign a LS instance to the one annotator. You can also use only one machine with different projects distributed across the annotators if it's fits your security reasons.

The script have the global and concrete arguments. The global ones define the settings path, project name in LS to interact with and port where the LS is deployed. Next you have to switch between the `get_annotation` and `deploy_project` modes.

The arguments for `get_annotation` are:
- `--output_file` - the path to store the data from LS in csv.
- `--with_comments_only` - if set, store the only examples where the file for comments (TextArea) is not empty.
- `--store_source_json` - if set, creates by the same path and stem a `.json` file that contains all information provided by the LS.

The arguments for `deploy_project` are:
- `--task_file` - path to the file with source data. It has to be a csv file where each row is one task or json file that fits the LS input data specification. The csv column with text to be annotated must be named as `text`. The input data should also contains `data_id` which is a unique id of each example.
- `--interface_config` - path to the file with annotation interface specification

Example of getting the annotation (assuming the settings file is in default path)

```bash
$ python src/scripts/label_studio_cli.py --port 9999 --project_name annotation_1 get_annotation --output_file annotated_data.csv 
```

Example of deploying project:

```bash
$ python src/scripts/label_studio_cli.py --port 9999 --project_name annotation_1 deploy_project --task_file data_for_annot.csv --interface_config labeling_interface.txt 
```
## aggregate_annotation.py

The script produce the aggregation for the overlapping annotation. After the collecting is done, the aggregation is applied. It's intended to use in bound with `label_studio_cli.py` script. Three aggregation types are available:
- equal - the final label is set iff all N annotators have the same annotation 
- maximum - the classic majority voting strategy
- soft - in the case when the multilabel is presented it brakes the labels into a list and chose labels that appear more then `num_annotators // 2 + 1` times.

The logic of the script is to collect csv files with annotation from one dir and produce one file with aggregation. The input files must be separable by `_` and have iteration number on position 3 (count from 0) and annotator name on last position. The input data also must have columns `data_id`, `text` and `annotation`.

On the output the script produces two files. One file contains the examples where the aggregation could produce a label (has `matched` in name) and one file where the aggregation fails to produce the label (has `unmatched` in name).

The arguments of the scripts are:
- `--src_dir` - the dir to scan for csv files.
- `--out_dir` - the output dir to store the result
- `--agg_method_to_match` - the aggregation method that will be used to separate matched examples
- `--postfix` - a postfix for result file names.
- `--use_only` - the iteration numbers to be used separated by semicolon.

Example of running the script
```bash
$ python src/scripts/aggregate_annotation.py --src_dir ./annot_data/test_part/ --out_dir ./annot_data/test_part/ --agg_method_to_match soft --postfix aggregated 
```

## `run_training.py`
The script runs the BERT-like model training procedure. The training is based on `Trainer` class of `transformers` library. The parameters and arguments are setting up by the Hydra configs in `./conf` directory. The main config is `conf/training_config.yaml` which has next parameters:

- `defaults` - base configuration sections including dataset, preprocessing, model and ClearML settings.
- `log_into_clearml` - boolean flag to enable logging to ClearML platform
- `seed` - random seed value for reproducibility
- `experiment_name` - name identifier for the current experiment run
- `output_dir` - directory path for saving model outputs and artifacts
- `save_total_limit` - maximum number of checkpoints to keep (null means no limit)
- `remove_checkpoints` - boolean flag to delete non-best checkpoints
- `logging_dir` - directory path for storing log files
- `truncation` - boolean flag to enable text truncation in tokenizer.
- `padding` - boolean flag to enable padding in tokenizer.
- `max_length` - maximum sequence length for text inputs in tokenizer.
- `test_run` - Boolean flag to indicate test execution mode. If True, run only on 16 random examples.
- `metric_for_best_model` - metric name used for model selection (must start with 'eval_')

The parameters for `conf/clearml/clearml.yaml`:

* `project` - The name of the project in ClearML platform.
* `task_name` - Task identifier composed of "bert_training-" prefix and experiment name variable.
* `task_type` - Specifies the type of task. Possible values: "training", "inference", "custom".
* `output_uri` - Base output URI path (empty string by default)
* `model_upload_dest` - Destination path for uploading trained models, combines ClearML output URI with "/models" suffix.

The parameters for `conf/dataset/base.yaml`:

- `dataset_id` - ClearML dataset id to download
- `download_from_clearml` - Flag indicating whether to download dataset from ClearML.
- `base_path` - root directory path with dataset files
- `label2id` - json file mapping text labels to numeric IDs.
- `text_col` - Column name containing input text data.
- `label_col` - Column name containing label data.
- `train_file` - CSV filename for training data.
- `test_file` - CSV filename for test data.
- `eval_file` - CSV filename for evaluation data (optional).
- `use_test_as_eval` - Flag to use test set for evaluation if no eval file provided.
- `sep` - Delimiter used in CSV files.

The parameters for `conf/model/bert-stage0.yaml`:

* `model_path` - Path to the pre-trained transformer model (DeepPavlov/rubert-base-cased)
* `freeze_encoder` - Boolean flag to freeze encoder layers during training 
* `epoch` - Number of training epochs
* `batch_size_train` - Batch size per device during training
* `batch_size_eval` - Batch size per device during validation
* `weight_decay` - Weight decay coefficient for regularization
* `learning_rate` - Learning rate for model optimization
* `save_total_limit` - Maximum number of checkpoints to keep

The preprocessing (`conf/preprocessing/preprocessing_v0.yaml`) is based on `textfab` module that has internal preprocess units and also can apply custom units. The internal units are placed under `primitives` key (see the `textfab` doc for all unit list), the custom ones are placed under `custom` key.

Example of running the training procedure:
```bash 
$ python src/scripts/run_training.py
```

Because it's based on Hydra, you can override the arguments during the run (other Hydra features also available). It especially useful when you only need to change the dataset and exp name.
```bash
$ python src/scripts/run_training.py experiment_name=exp_name dataset.base_path=/data/another_dataset
```

## `run_hpo_optuna.py`

This experimental script is for hyperparameter optimization using the Optuna. It's based on `run_training.py` script. In order to use it, you need to specify the parameter search space in lines 84-92. After that just run
```bash
$ python src/scripts/run_hpo_optuna.py
```

## `pretrain_bert.py` and `pretrain_gpt.py`

These scripts are for a pretraining the BERT-like and GPT like models with base config files `config/pretrain_config_bert.yaml` and `config/pretrain_config_gpt.yaml`, respectively. The structure generally resembles the `run_training.py` configs except for some hyperparameter that specific for the pretraining process. 

For BERT these are:
- mlm_probability - the probability of masked language modeling.
- gradient_accumulation_steps - perform optimization step after specified number of batches.

For GPT these are:
- max_len - the maximum length of the text fitted into model.

Example of running the training procedure:
```bash 
$ python src/scripts/pretrain_bert.py
$ python src/scripts/pretrain_gpt.py

```

## validate_puplished_models.py

The script allows to download published models and datasets from HuggingFace by given repo ids and makes validation in terms of metric values.
Published repo ids for each configuration (Antisuisidal and Presuisidal) presented in "Published models and datasets" bellow. It's important to use the proper dataset for the required model.

Example of validating antisuisidal model:
```bash 
$ python src/scripts/validate_published_models.py --dataset_hf_repo_id=psytechlab/antisuisidal_dataset --model_hf_repo_id=psytechlab/antisuisidal_model
```
        
Example of validating presuisidal model:      

```bash 
$ python src/scripts/validate_published_models.py --dataset_hf_repo_id=psytechlab/presuisidal_dataset --model_hf_repo_id=psytechlab/presuisidal_model
```


### Published models and datasets

|  Configuration |Dataset  |  Model |
|---|---|---|
| Antisuisidal  |  https://huggingface.co/datasets/psytechlab/antisuisidal_dataset | https://huggingface.co/psytechlab/antisuisidal_model  |   
| Presuisidal  |  https://huggingface.co/datasets/psytechlab/antisuisidal_dataset |  https://huggingface.co/psytechlab/presuisidal_model |   |   

# Parsers

This directory contains various parsers that was developed to acquire the data for the data annotation of presuisidal and antisuisidal signals. It's include parses for:

* vk.com
* 2ch.*
* archivach.*
* palata6.net
* pobedish.ru
* psyche.guru

# Metrics

This directory contains the implementation of different metrics, Currently it has:
- Self-BLEU - measure the lexical diversity.
- Distinct-N - measure the lexical diversity
- A collection fo pairwise distance based metrics for vectors:
  - Remote Clique
  - Chamfer distance
  - Minimum spanning tree dispersion
  - Sparseness
  - Span metric

# Contrib

This directory with the code from third-party developers, which was saved "as is" if it allows to work correctly.

- cartography - the code to perform dataset cartography.
- topmine - the algorithm to find the key phrases for text.
- v_info - compute v-usable information for the dataset.