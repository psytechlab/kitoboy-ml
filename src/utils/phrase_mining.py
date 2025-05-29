import re
from collections import Counter
from src.contrib.topmine.topmine import PhraseMining

def preprocess_input_text(text_list: list[str], stopwords=None):
    """Performs preprocessing on the input document.
    
    This func is a herritage of the original TopMine code.
    """
    documents = []
    for line in text_list:
        line_lowercase = line.lower()
        sentences_no_punc = re.split(r"[.,;!?]", line_lowercase)
        stripped_sentences = []
        for sentence in sentences_no_punc:
            stripped_sent = re.sub('[^А-Яа-я0-9]+', ' ', sentence).strip()
            if stopwords is not None:
                stripped_sent = ' '.join([word for word in stripped_sent.split() if word not in stopwords])
            stripped_sentences.append(stripped_sent)
        sentences_no_punc = stripped_sentences
        documents.append(sentences_no_punc)

    return documents

def mine_phrases(texts: list[str], 
                 min_support: int = 10, 
                 max_phrase_size: int = 100, 
                 alpha: int = 4, 
                 stop_words: list[str] | None = None, 
                 disable_preprocessing: bool = False):
    """Mine phrases using the TipMine algorithm.
    
    Args:
        texts(list[str]): The list of texts to be processed.
        min_support (int): Minimum support threshold which must be satisfied 
            by each phrase during frequent pattern mining.
        max_phrase_size (int): maximum allowed phrase size.
        alpha (int): threshold for the significance score calculation
            while merging the phrases in a clustering phase.
    Returns:
        (Counter): The counter object with all found phrases.
    """
    if not disable_preprocessing:
        processed_text = preprocess_input_text(texts, stop_words)
    else:
        processed_text = texts
    miner = PhraseMining(min_support=min_support, max_phrase_size=max_phrase_size, alpha=alpha)
    partitions, index_vocab = miner.fit(processed_text)
    phrases = [" ".join([index_vocab[y] for y in phrase_or_token]) for doc in partitions for phrase_or_token in doc if len(phrase_or_token) > 1]
    phrase_counter = Counter(phrases)
    return phrase_counter