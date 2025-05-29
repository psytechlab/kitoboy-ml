from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.contrib.cartography.selection.train_dy_filtering import compute_train_dy_metrics
from src.contrib.cartography.selection.train_dy_filtering import plot_data_map
from src.utils.text_features import apply_model

def get_predictions_from_all_checlpoins(
    path_to_checkpoints: Path, 
    texts: list[str], 
    tokenizer: AutoTokenizer | None = None,
    device="cuda"):
    """Compute the preds for all checkpoints created by transformers Trainer.
    
    Args:
        path_to_checkpoints(Path): path to the folder with Trainer checkpoints
        texts (list[str]): texts that was used in training split
        tokenizer (AutoTokenizer): model tokenizer. It can be ommited
            if the checkpoints was saved with it.
        device(str): the compute device for the model
    Returns:
        (list[list[list[float]]]): train logits for each checkpoint [num_epoch x num_examples x num_classes]
        
    """
    if isinstance(path_to_checkpoints, str):
        path_to_checkpoints = Path(path_to_checkpoints)
    preds_across_epoch = []
    for model_dir in tqdm(path_to_checkpoints.iterdir()):
        if not model_dir.is_dir() or "best_model" in model_dir.name:
            continue
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(path_to_checkpoints)
            except OSError:
                raise ValueError("The tokenizer cannot be loaded from the model dir. Specify the tokenizer trough the argument.")
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.to(device)
        preds = apply_model(texts, model, tokenizer, keep_logits=True)
        preds_across_epoch.append(preds)
        model.to("cpu")
        del model
    preds_across_epoch = [x.tolist() for x in preds_across_epoch]
    return preds_across_epoch
    
def compute_trainig_dynamics_offline(checkpoint_path: Path, texts: list[str], labels: list[int], tokenizer: AutoTokenizer | None =None, burn_out: int=100, include_ci:bool = True):
    """Compute training dynamics on the available checkpoints.
    
    The checkpoints has to be computed on equal distance on the set.
    Default is one epoch.
    The texts are from the same set on the the trainin was done.
    
    This function is an interinterface for the official code for cartography computing.
    
    Args:
        checkpoint_path(Path): Path to the Trainer checkpoints
        texts(list[str]): texts from the training set
        labels(list[int]): corresponding labels as intgers
        tokenier(AutoTokenizer): if the model checkpoints don't have tokenizer,
        you have to specify it in this parameter.
        burn_out (int):  Number of epochs for which to compute train dynamics.
        include_ci(bool): Whether to compute the confidence interval for variability.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: training dynamics and general metrics in form
        of DataFrame.
    """

    training_dynamics = {}
    # this is a wrapper for the compute_train_dy_metrics func args
    args = OmegaConf.create({"burn_out" : burn_out, "include_ci" : include_ci})
    
    preds_across_epoch = get_predictions_from_all_checlpoins(checkpoint_path, texts, tokenizer)
    for idx, l in enumerate(labels):
        obj = {
            "gold": l,
            "logits": [preds_across_epoch[x][idx] for x in range(len(preds_across_epoch))]
        }
        training_dynamics[idx] = obj
    
    training_dynamic_metrics, common_metrics  = compute_train_dy_metrics(training_dynamics, args)
    return training_dynamic_metrics, common_metrics