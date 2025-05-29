"""The script with functions to work with clearml."""
import os
import clearml
from pathlib import Path
from shutil import copytree, rmtree


def upload_dataset_clearml(
    dataset_name: str,
    path: str,
    output_uri: str,
    dataset_project: str = "kitoboy",
):
    """Upload dataset to ClearML server.

    Args:
        dataset_name (str): Dataset name for ClearML
        path (str): Local path to the dataset.
        dataset_project (str, optional): The project for the dataset. Defaults to "kitoboy".
        output_uri (_type_, optional): Where to store the dataset physically.

    Raises:
        ValueError: Local path doesn't exist.
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.exists():
        raise ValueError("Path doesn't exists")
    ds = clearml.Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project,
        output_uri=output_uri,
    )
    ds.add_files(path=path)
    ds.upload()
    ds.finalize()


def get_clearml_dataset_by_id(dataset_id: str) -> Path:
    """Download the dataset from Clearml.

    Args:
        dataset_id (str): The dataset id in Clearml.

    Returns:
        Path: The local path to the dataset.
    """
    return Path(clearml.Dataset.get(dataset_id).get_local_copy())


def download_dataset_clearml(dataset_id: str, path: tuple | Path) -> Path:
    """Download the dataset from clearml and placed in desired dir.

    Args:
        dataset_id (str): The clearml dataset id.
        path (tuple | Path): Local path.

    Returns:
        Path: Path to the dataset
    """
    if isinstance(path, str):
        path = Path(path)
    if path.exists():
        raise
    dataset_path = get_clearml_dataset_by_id(dataset_id)
    os.system(f"cp -r {str(dataset_path)} {str(path)}")
    return path


def save_model_clearml(
    model_name: str, 
    model_local_path: str, 
    output_uri: str,
    upload_destination: str,
    auto_delete_file: bool = False, 
    task_type: str = "custom"
) -> None:
    """Save model to clearml.

    Args:
        model_name (str): Model name in clearml
        model_local_path (str): Local path to model for saving.
        auto_delete_file (bool): whether to delete model checkpoints locally after uploading to the clearml.
    """
    task = clearml.Task.init(
        project_name="kitoboy",
        task_name=model_name,
        task_type=task_type,
        output_uri=output_uri,
    )
    try:
        out_model = clearml.OutputModel(task=task, name=model_name, framework="Torch")
        out_model.set_upload_destination(upload_destination)
        out_model.update_weights(
            weights_filename=model_local_path, auto_delete_file=auto_delete_file
        )
    except Exception as e:
        print(e)
    finally:
        task.close()
        

def download_model_from_clearml(model_id: str, target_path: str | Path | None = None, override: bool = False):
    """Download model from clearml by its id.

    Args:
        model_id (str): The model id from Clearml.
        target_path (str | Path | None, optional): Copy model to this path if provided. Defaults to None.
        override (bool, optional): Override the target path. Defaults to False.

    Raises:
        ValueError: Raise if target path exists and override option is false.

    Returns:
        Path: Path with the model.
    """ 
    if isinstance(target_path, str):
        target_path = Path(target_path)
    model_path = clearml.InputModel(model_id).get_local_copy(extract_archive=True)
    if target_path is None:
        return model_path
    if target_path.exists() and override:
        rmtree(target_path)
        copytree(model_path, target_path)
    elif not target_path.exists():
        copytree(model_path, target_path)
    else:
        raise ValueError(f"Path {target_path} exists.")
    return target_path
