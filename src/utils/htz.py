from collections import Counter
from math import log
from nltk import word_tokenize
from tqdm import tqdm

def calculate_idf(tokenized_texts: list[list[str]]):
    idf = Counter()
    for t in tokenized_texts:
        for w in set(t):
            idf[w] += 1
    for w in idf:
        idf[w] = log(len(tokenized_texts)/idf[w], 2)
    return idf

def heaviside(x: int | float):
    if x< 0:
        return 0
    elif x == 0:
        return 0.5
    else:
        return x

def calculate_htz(text_inclass: list, text_outclass: list):
    metric = []
    t1 = calculate_idf(text_inclass)
    t2 = calculate_idf(text_outclass)
    for w in t1:
        delta = t2[w] - t1[w]
        metric.append( (w, delta * heaviside(delta)) )
    metric.sort(key = lambda x: x[1], reverse=True)

    return metric

def get_htz(df, text_col: str, label_col: str, top_k: int = 30, tokenizer=None):
    df = df.reset_index()
    texts = df[text_col].to_list()
    if tokenizer is None:
        texts = [word_tokenize(t) for t in texts]
    else:
        texts = tokenizer(texts)

    lexicon = {l: [] for l in df[label_col].to_list()}
    for label in tqdm(lexicon):
        print(label)
        in_class = [texts[i] for i in df[df[label_col] == label].index]
        out_class = [texts[i] for i in df[df[label_col] != label].index]

        lexicon[label] = [x[0] for x in calculate_htz(in_class, out_class)][:top_k]
    #FIXME: check that all word lists are equal
    return lexicon#pd.DataFrame(lexicon)