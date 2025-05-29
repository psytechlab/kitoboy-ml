"""Script for data enrichment by heuristics."""
from uuid import uuid4
from src.utils.clearml_utils import download_dataset_clearml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import torch.nn.functional as F
from textfab.fabric import Fabric

import re
import torch
import numpy as np


RE_ENG_WORDS = re.compile("[A-Za-z]")

EMO_LABELS = ["no_emotion", "joy", "sadness", "surprise", "fear", "anger"]

PERSONAL_PRONOUNS = {
    "я",
    "мы",
    "себя",
    "себе",
    "собой",
    "собою",
    "нас",
    "нам",
    "нами",
    "мой",
    "мне",
    "меня",
    "мной",
}

NOT_PERSONAL_PRONOUNS = {
    "ты",
    "вы" "он",
    "она",
    "они",
    "им",
    "его",
    "ему",
    "ее",
    "её",
    "ей",
    "еи",
    "их",
    "им",
    "ими",
    "вас",
    "вам",
    "тебе",
}

MARRIEDS_ROLES = {
    "муж",
    "мужу",
    "мужа",
    "муже",
    "супруг",
    "супругу",
    "супругом",
    "жена",
    "жёнушка",
    "жене",
    "жёнушке",
    "женой",
    "жёнушкой",
    "супруге",
    "супругой",
}

PARENTS_ROLES = {
    "родич",
    "родственник",
    "родитель",
    "родител",
    "мать",
    "матери",
    "матерь",
    "мама",
    "маме",
    "маму",
    "мамой",
    "мамин",
    "мачеха",
    "матушка",
    "отец",
    "отца",
    "отцу",
    "отцом",
    "папа",
    "папе",
    "папой",
    "батя",
    "отчим",
    "папаня",
    "кормилец",
    "бате",
    "отчиму",
    "папане",
    "кормильцу",
    "батей",
    "отчимом",
    "папаней",
    "кормильцем",
}

FAMILY_ROLES = {
    "родственник",
    "брат",
    "брату",
    "брате",
    "братом",
    "сестра",
    "систр",
    "сестре",
    "систе",
    "бабушка",
    "баба",
    "бубуля",
    "бабушку",
    "бабe",
    "бубулe",
    "дедушка",
    "дед",
    "дедуля",
    "дедушке",
    "деду",
    "дедуле",
    "дедушкой",
    "дедом",
    "дедулей",
    "дядя",
    "дядька",
    "дядюшка",
    "дяди",
    "дядьки",
    "дядюшке",
    "тетя",
    "тетк",
    "тетей",
    "теткой",
}

FREIND_LOVER_ROLES = {
    "друз",
    "друг",
    "френд",
    "подруг",
    "девушка",
    "парень",
    "парню",
    "парня",
    "любовник",
    "любовниц",
}

emo_tokenizer = AutoTokenizer.from_pretrained(
    "cointegrated/rubert-tiny2-cedr-emotion-detection"
)
emo_model = AutoModelForSequenceClassification.from_pretrained(
    "cointegrated/rubert-tiny2-cedr-emotion-detection"
)
emo_model.eval()
emo_model = emo_model.to("cuda")

sui_tokenizer = AutoTokenizer.from_pretrained("astromis/presuisidal_rubert")
sui_model = AutoModelForSequenceClassification.from_pretrained(
    "astromis/presuisidal_rubert"
)
sui_model.eval()
sui_model = sui_model.to("cuda")

sent_tokenizer = AutoTokenizer.from_pretrained(
    "sismetanin/rubert-ru-sentiment-rusentiment"
)
sent_model = AutoModelForSequenceClassification.from_pretrained(
    "sismetanin/rubert-ru-sentiment-rusentiment"
)
sent_model.eval()
sent_model = sent_model.to("cuda")


def extract_orig_id(sent_id: str):
    """Extract text id from sentence id.

    Args:
        sent_id (str): Sentence id.

    Returns:
        str: text id.
    """
    orig_id = sent_id.split("-")[0]
    return orig_id


def simple_tokenizer(sentence: str, filter_punct_tokens: bool = True):
    """Tokenize text with simple heuristics.

    Args:
        sentence (str): The source sentence.
        filter_punct_tokens (bool, optional): If True, filter the punctuations. Defaults to True.

    Returns:
        list[str]: tokenzied text.
    """
    tokens = []
    if filter_punct_tokens:
        for token in sentence.split():
            if any([token.isalnum(), token.isalpha(), token.isalpha()]):
                tokens.append(token)
            else:
                for l in token:
                    if any([l.isalnum(), l.isalpha(), l.isdigit()]):
                        tokens.append(token)
                        break
    else:
        tokens = sentence.split()

    return tokens


def get_n_of_tokens(sentence: str, filter_punct_tokens: bool = True):
    """Get number of tokens from text.

    Args:
        sentence (str): The source text.
        filter_punct_tokens (bool, optional): If True, filter the punctuations. Defaults to True.

    Returns:
        int: number of tokens.
    """
    n_tokens = len(simple_tokenizer(sentence, filter_punct_tokens))
    return n_tokens


def has_english(s: str):
    """If text has english texts by latin symbols.

    Args:
        s (str): The source text

    Returns:
        bool: True if eng text is presented, False otherwise.
    """
    res = re.search(RE_ENG_WORDS, s)
    res = False if res is None else True
    return res


def does_text_has_tokens(text: str, tokens: set):
    """Check if text has specified tokens.

    Args:
        text (str): The source text.
        tokens (set): Token set to check.

    Returns:
        bool: True if at least one token found, False otherwise.
    """
    uniq_tokens = set(text.lower().split())
    return bool(len(uniq_tokens & tokens))


def has_personal_pronouns(sent: str, pronouns=PERSONAL_PRONOUNS):
    return does_text_has_tokens(sent, pronouns)


def has_not_personal_pronouns(sent: str, pronouns=NOT_PERSONAL_PRONOUNS):
    return does_text_has_tokens(sent, pronouns)


def has_marrieds_roles(sent: str, roles=MARRIEDS_ROLES):
    return does_text_has_tokens(sent, roles)


def has_parents_roles(sent: str, roles=PARENTS_ROLES):
    return does_text_has_tokens(sent, roles)


def has_family_roles(sent: str, roles=FAMILY_ROLES):
    return does_text_has_tokens(sent, roles)


def has_friend_roles(sent: str, roles=FREIND_LOVER_ROLES):
    uniq_tokens = set(sent.lower().split())
    for r in roles:
        for t in uniq_tokens:
            if r in t and "френдзон" not in t and "френдли" not in t:
                return True
    return False


def has_friendzone(sent: str, roles=FREIND_LOVER_ROLES):
    uniq_tokens = set(sent.lower().split())
    for r in roles:
        for t in uniq_tokens:
            if "френдзон" in t or "френд-зон" in t:
                return True
    return False


@torch.no_grad()
def apply_emotion_model(
    text_list: list[str], batch_size: int = 32, labels: list[str] = EMO_LABELS
):
    """Apply emotion model to texts.

    Args:
        text_list (list[str]): The list of texts.
        batch_size (int, optional): Batch size. Defaults to 32.
        labels (list, optional): Emotional labels.. Defaults to EMO_LABELS.

    Returns:
        tuple[list[str], list[float]]: Predictions with its scores.
    """
    argmax_preds = []
    multiple_probs = []
    for i in tqdm(range(0, len(text_list), batch_size)):

        inputs = emo_tokenizer(
            text_list[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = inputs.to(emo_model.device)

        outputs = emo_model(**inputs)

        pred = torch.nn.functional.sigmoid(outputs.logits)
        argmax_preds.extend([labels[p.argmax()] for p in pred])

        emotions_list = [
            {labels[i]: round(p.tolist()[i], 3) for i in range(len(p.tolist()))}
            for p in pred
        ]

        multiple_probs.extend(emotions_list)
    return argmax_preds, multiple_probs


def apply_model(
    text_list: list[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    keep_logits: bool = False,
    use_softmax: bool = True
):
    """Apply common HF model to texts.

    Args:
        text_list (list[str]): The list of text.
        model (AutoModelForSequenceClassification): HF BERT model.
        tokenizer (AutoTokenizer): Tokenizer to usr.
        batch_size (int, optional): Batch size. Defaults to 32.
        keep_logits(bool, optional): if True, the logits return as full matrix
        use softmax(bool, optional): if True, apply softmax to the logits

    Returns:
        (np.array, np.array): Predicted classes and scores.
    """
    pred_classes = []
    pred_scores = []
    for i in tqdm(range(0, len(text_list), batch_size)):
        tokenized_text = tokenizer(
            text_list[i : i + batch_size],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        tokenized_text = tokenized_text.to("cuda")
        with torch.no_grad():
            prediction = model(**tokenized_text).logits
        if use_softmax:
            prediction = F.softmax(prediction, dim=1)
        pred_classes.append(prediction.argmax(dim=1))
        if not keep_logits:
            pred_scores.append(prediction.max(dim=1).values)
        else:
            pred_scores.append(prediction)
            
    if not keep_logits:
        pred_scores = np.hstack([x.detach().cpu().numpy() for x in pred_scores])
    else:
        pred_scores = np.vstack([x.detach().cpu().numpy() for x in pred_scores])
    pred_classes = np.hstack([x.detach().cpu().numpy() for x in pred_classes])
    return pred_classes, pred_scores


def preprocess(corp):
    """Preprocess text for presuicidal model.

    Args:
        corp (list[str]): List of texts to process.

    Returns:
        list[str]: Processed texts.
    """
    con = Fabric(
        ["remove_punct", "lower_string", "swap_enter_to_space", "collapse_spaces"]
    )
    corp = list(map(lambda x: re.sub(r"<emoji>.+</emoji>", "", x), corp))
    corp = list(map(lambda x: re.sub("[A-Za-z]+", "", x), corp))
    corp = con(corp)
    return corp


def featurize_dataframe(df_to_featurize, text_col="sent"):
    """Get all heuristrcs and models predictions for DataFrame.

    Args:
        df_to_featurize (pd.DataFrame): DataFrame to featurize.
        text_col (str, optional): Name of column to use. Defaults to "sent".

    Returns:
        pd.DataFrame: Featurized dataframe.
    """
    df_to_featurize["has_english"] = df_to_featurize[text_col].apply(has_english)
    df_to_featurize["n_of_word_tokens"] = df_to_featurize[text_col].map(get_n_of_tokens)
    df_to_featurize["n_of_all_tokens"] = df_to_featurize[text_col].apply(
        lambda x: get_n_of_tokens(x, filter_punct_tokens=False)
    )

    argmax_preds, multiple_probs = apply_emotion_model(
        df_to_featurize[text_col].tolist()
    )
    df_to_featurize["emotion_argmax_label"] = argmax_preds

    for em in EMO_LABELS:
        df_to_featurize[em] = [i[em] for i in multiple_probs]

    preds = apply_model(
        preprocess(df_to_featurize.sent.astype(str).to_list()), sui_model, sui_tokenizer
    )
    df_to_featurize["presuicidal_model_pred"] = preds

    preds = apply_model(
        df_to_featurize.sent.astype(str).to_list(), sent_model, sent_tokenizer
    )
    preds = [sent_model.config.id2label[x] for x in preds]
    df_to_featurize["sa_label"] = preds

    df_to_featurize["uncased_sent"] = df_to_featurize[text_col].map(
        lambda x: x.lower().replace("ё", "е")
    )

    df_to_featurize["has_pers_prns"] = list(
        map(has_personal_pronouns, df_to_featurize["uncased_sent"])
    )
    df_to_featurize["has_not_pers_prns"] = list(
        map(has_not_personal_pronouns, df_to_featurize["uncased_sent"])
    )
    df_to_featurize["has_marrieds_roles"] = list(
        map(has_marrieds_roles, df_to_featurize["uncased_sent"])
    )
    df_to_featurize["has_parents_roles"] = list(
        map(has_parents_roles, df_to_featurize["uncased_sent"])
    )
    df_to_featurize["has_family_roles"] = list(
        map(has_family_roles, df_to_featurize["uncased_sent"])
    )
    df_to_featurize["has_friend_roles"] = list(
        map(has_friend_roles, df_to_featurize["uncased_sent"])
    )
    df_to_featurize["has_friendzone"] = list(
        map(has_friendzone, df_to_featurize["uncased_sent"])
    )
    return df_to_featurize
