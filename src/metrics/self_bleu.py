"""based on https://github.com/geek-ai/Texygen/blob/master/utils/metrics/SelfBleu.py"""

import nltk
import os
from multiprocessing import Pool
from nltk.translate.bleu_score import SmoothingFunction
import random

def calc_bleu(reference: list[list[str]], hypothesis:list[str], weight:list[float]):
    """Wrap the BLEU calculation to execute in pool.

    Args:
        reference (list[list[str]]): List with tokenized reference texts.
        hypothesis (list[str]): A tokenized hypothesis text.
        weight (list[float]): A list of weights for ngrams.

    Returns:
        float | list[float]:  The sentence-level BLEU score. Returns a list if multiple weights were supplied.

    """
    return nltk.translate.bleu_score.sentence_bleu(reference, hypothesis, weight,
                                                   smoothing_function=SmoothingFunction().method1)

def calculate_self_bleu(texts: list[list[str]], sample_size: int | None = 500, do_average=True, ngram: int = 3):
    """Calculate Self-BLEU score for a given tokenized texts.

    Args:
        texts (list[list[str]]): A list of tokenized text.
        sample_size (int | None, optional): Amount of sample to pick. Reduce the computition time. Defaults to 500.
        do_average (bool, optional): If True, average all individual scores, otherwise return list a scores. Defaults to True.
        ngram (int, optional): A max ngram magnitude to use. Defaults to 3.

    Returns:
        float|list[float]: The self-bleu value.
    """
    if sample_size is not None:
        texts = random.sample(texts, sample_size)
    weight = tuple(1. / ngram for _ in range(ngram))
    pool = Pool(os.cpu_count())
    result = []
    for index, hypothesis in enumerate(texts):
        other = texts[:index] + texts[index+1:]
        result.append(pool.apply_async(calc_bleu, args=(other, hypothesis, weight)))
    pool.close()
    pool.join()
    res = [i.get() for i in result]
    if do_average:
        return sum(res)/len(res)
    return res
    