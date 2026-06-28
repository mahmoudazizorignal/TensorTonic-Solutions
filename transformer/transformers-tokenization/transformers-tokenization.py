import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 4
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

        for id, token in enumerate([
            self.pad_token,
            self.unk_token,
            self.bos_token,
            self.eos_token,
        ]):
            self.word_to_id[token] = id
            self.id_to_word[id] = token
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        words = []
        for text in texts:
            words.extend(text.lower().strip().split(" "))
        words = sorted(list(set(words)))
        self.vocab_size += len(words)
        for id, word in enumerate(words):
            self.word_to_id[word] = id + 4
            self.id_to_word[id + 4] = word
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        words = text.lower().strip().split(" ")
        return [self.word_to_id.get(word, self.word_to_id[self.unk_token]) for word in words if len(word) > 0]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join([self.id_to_word.get(id, self.unk_token) for id in ids]).strip()
