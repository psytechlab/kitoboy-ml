"""https://github.com/neural-dialogue-metrics/Distinct-N/blob/master/distinct_n/metrics.py"""
from nltk import ngrams

def distinct_n_sentence_level(sentence: list[str], n: int):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences: list[list[str]], n: int, do_average: bool =True):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    sentencewise_results = [distinct_n_sentence_level(sentence, n) for sentence in sentences]
    if do_average:
        return sum(sentencewise_results) / len(sentences)
    return sentencewise_results
