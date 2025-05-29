"""Original code: https://github.com/kawine/dataset_difficulty/
Paper: https://arxiv.org/pdf/2110.08420"""

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from tokenizers.pre_tokenizers import Whitespace
from src.utils.text_features import apply_model

def v_entropy(texts, labels, model, tokenizer, batch_size=64):
    """
    Calculate the V-entropy (in bits) on the data given in data_fn. This can be
    used to calculate both the V-entropy and conditional V-entropy (for the
    former, the input column would only have null data and the model would be
    trained on this null data).

    Args:
        texts: texts data
        labels: corresponding labels
        model: HF Model for Sequence cls
        tokenizer: corresponding tokenizer
        batch_size: data batch_size

    Returns:
        Tuple of (V-entropies, correctness of predictions, predicted labels).
        Each is a List of n entries (n = number of examples in data_fn).
    """
    if isinstance(labels, list):
        labels = np.array(labels)
    preds, scores = apply_model(texts, model, tokenizer, batch_size=batch_size, keep_logits=True)
    pred_scores_for_true = np.take_along_axis(scores, labels.reshape(-1, 1), axis=1).flatten()
    entropies = -1 * np.log2(pred_scores_for_true)
    is_correct = labels == preds
    return entropies, is_correct, preds

def v_info(df, model, null_model, tokenizer):
    """
    Calculate the V-entropy, conditional V-entropy, and V-information on the
    data in data_fn. Add these columns to the data in data_fn and return as a 
    pandas DataFrame. This means that each row will contain the (pointwise
    V-entropy, pointwise conditional V-entropy, and pointwise V-info (PVI)). By
    taking the average over all the rows, you can get the V-entropy, conditional
    V-entropy, and V-info respectively.

    Args:
        df: dataframe with 'text' as data and 'label' columns 
        model: HF Model for Sequence cls trained on normal data
        null_data:  HF Model for Sequence cls trained on null data (empty string)
        tokenizer: corresponding tokenizer

    Returns:
        Pandas DataFrame of the data in data_fn, with the three additional 
        columns specified above.
    """
    
    df['H_yb'], _, _ = v_entropy_([""]*len(df), df.label.to_list(), null_model, tokenizer) 
    df['H_yx'], df['correct_yx'], df['predicted_label'] = v_entropy_(df.text.to_list(), df.label.to_list(), model, tokenizer)
    df['pvi'] = df['H_yb'] - df['H_yx']
    return df

def find_annotation_artefacts(texts, labels, model, tokenizer, min_freq=5, pre_tokenize=True):
    """
    Find token-level annotation artefacts (i.e., tokens that once removed, lead to the
    greatest decrease in PVI for each class).

    Args:
        texts: texts data
        labels: corresponding labels
        model: HF Model for Sequence cls trained on normal data
        tokenizer: corresponding tokenizer
        min_freq: minimum number of times a token needs to appear (in a given class' examples)
            to be considered a potential partefact
        pre_tokenize: if True, do not consider subword-level tokens (each word is a token)

    Returns:
        A pandas DataFrame with one column for each unique label and one row for each token.
        The value of the entry is the entropy delta (i.e., the drop in PVI for that class if that
        token is removed from the input). If the token does meet the min_freq threshold, then the
        entry is empty.
    """
    label_set = list(set(labels)) # assume labels are numbers
    token_entropy_deltas = { l : defaultdict(list) for l in label_set }
    all_tokens = set()

    pre_tokenizer = Whitespace()

    # get the PVI for each example
    print("Getting conditional V-entropies ...")
    entropies, _, _ = v_entropy_(texts, labels, model, tokenizer)

    print("Calculating token-wise delta for conditional entropies ...")
    pertrubed_texts = []
    deleted_tokens_log = []
    for i, example in tqdm(enumerate(texts)):
        if pre_tokenize:
            tokens = [ t[0] for t in pre_tokenizer.pre_tokenize_str(example) ]
        else:
            tokens = tokenizer.tokenize(example)
        # create m versions of the input in which one of the m tokens it contains is omitted
        curr_tok = []
        for j, tok in enumerate(tokens):
            # create new input by omitting token j
            pertrubed_texts.append(tokenizer.convert_tokens_to_string(tokens[:j] + tokens[j+1:]))
            curr_tok.append(tok)
            all_tokens.add(tok)
        deleted_tokens_log.append(curr_tok)

    # get the predictions 
        
    preds, scores = apply_model(pertrubed_texts, model, tokenizer, keep_logits=True)
    labels_for_pertrubed_texts = []
    for i,x in enumerate(deleted_tokens_log):
        labels_for_pertrubed_texts.extend([labels[i]]*len(x))
        
    token_entropies, _, _ = v_entropy_(pertrubed_texts, labels_for_pertrubed_texts, model, tokenizer)
    
    entropies_for_pertrubed_texts = []
    for i,x in enumerate(deleted_tokens_log):
        entropies_for_pertrubed_texts.extend([entropies[i]] * len(x))
    entropy_deltas = token_entropies - entropies_for_pertrubed_texts
    for token, entropy_val, label in zip([x for y in deleted_tokens_log for x in y], entropy_deltas, labels_for_pertrubed_texts):
        token_entropy_deltas[label][token].append(entropy_val)

    torch.cuda.empty_cache()

    total_freq = { t : sum(len(token_entropy_deltas[l][t]) for l in label_set) for t in all_tokens }
    # average over all instances of token in class
    for label in label_set:
        for token in token_entropy_deltas[label]:
            if total_freq[token] > min_freq:
            	token_entropy_deltas[label][token] = np.nanmean(token_entropy_deltas[label][token]) 
            else:
                token_entropy_deltas[label][token] = np.nan

    table = pd.DataFrame.from_dict(token_entropy_deltas).reset_index().rename({"index":"token"}, axis=1)
    return table