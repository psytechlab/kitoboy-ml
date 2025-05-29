import re
from textfab.base import ProcessUnit


class normalize_punctuation(ProcessUnit):
    """Make sentence separated punctuation in the right way.

    It corrects the position of the set {. , ? !} in such a way
    that they followed by next word without space and after them
    the the space follows.  Comma is exception.
    
    Ex.:
    "Nice day .Like Summer" - wrong.
    "Nice day.Like Summer" - wrong.
    "Nice day. Like Summer" - correct.
    """
    def process(self, text:str) -> str:
        return re.sub(r" ?([\.,\?\!]) ?", r"\g<1> ", text)

    def __str__(self):
        return "normalize_punctuation"

    
class collapse_multiple_punctuation(ProcessUnit):
    """Collapse sequence of {. , ? ! ( ) } into one symbol."""
    def process(self, text: str) -> str:
        return re.sub(r"([\.,\?\!\(\)])+", r"\g<1>", text)

    def __str__(self):
        return "collapse_multiple_punctuation"
    
    
class replace_ru_e(ProcessUnit):
    """Replaces 'ё' with 'е' in ru string. """
        
    def process(self, text):
        return text.replace('ё', 'е')
    
    def __str__(self):
        return "replace_ru_e"