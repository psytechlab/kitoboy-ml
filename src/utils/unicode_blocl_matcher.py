from __future__ import annotations

import re
from collections import defaultdict


class UnicodeBlockMatcher:
    """The class provides a function to map letters and symbols
    encoded in unicode into named blocks that was defined by
    Unicode Standard. 

    The blocks are being updated. You can take the latest
    version of the blocks by this link:
    https://www.unicode.org/Public/UCD/latest/ucd/Blocks.txt.

    More info: https://www.unicode.org/faq/blocks_ranges.html
    """
    def __init__(self, block_map_path: str):
        with open(block_map_path) as f:
            unicode_blokcs = f.read().split("\n")
        self.regex = []
        self.block_name = []
        for block_range, block_name in [x.split("; ") for x in unicode_blokcs]:
            block_range = block_range.split("..")
            block_range = rf"[\u{block_range[0]}-\u{block_range[1]}]"
            self.regex.append(re.compile(block_range))
            self.block_name.append(block_name)

    def match_symbol(self, symbol: str) -> str:
        """Match one symbol with the blocks.

        Args:
            symbol (str) : Single symbol. The tale will be ignored.
        Returns:
            str: The block name. If block wasn't found, than 'Unknown block'
        """
        for i, regex in enumerate(self.regex):
            if regex.match(symbol) is not None:
                return self.block_name[i]
        return "Unknown block"
        
    def map_text(self, texts: str | list[str]) -> dict[str, list[str]]:
        """Map all symbols from text into block names."""
        if isinstance(texts, str):
            texts = [texts]
        symbol_set = list(set(" ".join(texts)))
        block_names = [self.match_symbol(x) for x in symbol_set]
        result_map = defaultdict(list)
        for bname, sym in zip(block_names, symbol_set):
            result_map[bname].append(sym)
        return result_map
