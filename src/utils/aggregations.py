import pandas as pd

def max_aggregation(series: pd.Series,
                    num_overlaps: int) -> [str, None]:
    """Aggregates labels by majority.
    Args:
        series: Series with labels.
        num_overlaps: Number of overlaps.
    Returns:
        max_value: The label chosen by majority.
    """
    labels = list(series.values)
    labels = list(filter(lambda x: x != None, labels))
    max_value = max(labels, key=labels.count)
    max_count = labels.count(max_value)
    if max_count >= num_overlaps // 2 + 1:
        return max_value
    return None

def eq_aggregation(series: pd.Series,
                    num_overlaps: int) -> [str, None]:
    """Aggregates labels by majority.
    Args:
        series: Series with labels.
        num_overlaps: Number of overlaps.
    Returns:
        max_value: The label chosen by majority.
    """
    labels = list(series.values)
    if labels.count(labels[0]) == len(labels):
        return labels[0]
    return None

def soft_aggregation(series: pd.Series,
                    num_overlaps: int,
                    multilabel_split: str = ";") -> [str, None]:
    """Aggregates labels by majority.
    Args:
        series: Series with labels.
        num_overlaps: Number of overlaps.
    Returns:
        max_value: The label chosen by majority.
    """
    labels = list(series.values)
    labels = [y for x in labels for y in x.split(multilabel_split)]
    final = set()
    for l in labels:
        if labels.count(l) >=  num_overlaps // 2 + 1:
            final.add(l)
    final = list(final)
    final.sort()
    if len(final) > 0:
        return ";".join(final)
    return None
    