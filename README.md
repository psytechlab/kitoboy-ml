# kitoboy-ml



# Scripts

## label_studio_cli.py

The script is designed to interact with Label Studio. It allows you to download or deploy annotation data.

You fabe have to provide the settings file in which the host with Label Studio is defined as well as a dictionary with ports as keys and corresponding access tokens as values.

Example of getting the annotatinon (assuming the settings file is in default path)

```bash
$ python src/secipts/label_studio_cli.py --port 9999 --project_name annotation_1 get_annotation --output_file annotated_data.csv 
```

Example of deploying project:

```bash
$ python src/secipts/label_studio_cli.py --port 9999 --project_name annotation_1 deploy_project --task_file data_for_annot.csv --interface_config labeling_interface.txt 
```


## run_training.py
Before running traininf script you need to setupt ```conf/training_config.yaml```.

Example of training model:
```bash 
$ python src/scripts/run_training.py
```

## validate_puplished_models.py

The script allows to download published models and datasets from HuggingFace by given repo ids and makes validation.
Puplished repo ids for each cjnfiguration (Antisuisidal and Presuisidal) presented in "Published models and datasets" bellow. It's important to use the proper dataset for the required model.



Example of validating antisuisidal model:
```bash 
$ python src/scripts/validate_published_models.py --dataset_hf_repo_id=psytechlab/antisuisidal_dataset --model_hf_repo_id=psytechlab/antisuisidal_model
```
        
Example of validating presuisidal model:      

```bash 
$ python src/scripts/validate_published_models.py --dataset_hf_repo_id=psytechlab/presuisidal_dataset --model_hf_repo_id=psytechlab/presuisidal_model
```


# Published models and datasets

|  Configuration |Dataset  |  Model |
|---|---|---|
| Antisuisidal  |  https://huggingface.co/datasets/psytechlab/antisuisidal_dataset | https://huggingface.co/psytechlab/antisuisidal_model  |   
| Presuisidal  |  https://huggingface.co/datasets/psytechlab/antisuisidal_dataset |  https://huggingface.co/psytechlab/presuisidal_model |   |   