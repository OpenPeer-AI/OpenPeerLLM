import json
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import regex as re

class OpenPeerTokenizer:
    """Simple tokenizer implementation for testing"""
    
    def __init__(self, unk_token="<|endoftext|>", 
                 bos_token="<|endoftext|>",
                 eos_token="<|endoftext|>",
                 pad_token="<|endoftext|>"):
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.eos_token_id = 0
        
        # Get vocabulary
        self.vocab = self._get_default_vocab()
        self.vocab_size = len(self.vocab)
        
    def _get_default_vocab(self) -> Dict[str, int]:
        """Get a basic default vocabulary"""
        vocab = {}
        # Add special tokens
        vocab[self.unk_token] = 0
        vocab[self.pad_token] = 1
        vocab["<|mask|>"] = 2
        
        # Add basic ASCII characters and common words
        for i in range(32, 127):
            vocab[chr(i)] = len(vocab)
        
        # Add some common words
        common_words = ["the", "be", "to", "of", "and", "a", "in", "that", "have"]
        for word in common_words:
            vocab[word] = len(vocab)
            
        return vocab
        
    def __call__(self, text: Union[str, List[str]], **kwargs) -> Dict[str, List[int]]:
        """Tokenize text"""
        if isinstance(text, str):
            # Split into words and characters
            tokens = []
            for word in text.split():
                # Add word if in vocab, otherwise split into characters
                if word in self.vocab:
                    tokens.append(self.vocab[word])
                else:
                    for char in word:
                        tokens.append(self.vocab.get(char, self.vocab[self.unk_token]))
        else:
            tokens = []
            for t in text:
                word_tokens = []
                for word in t.split():
                    if word in self.vocab:
                        word_tokens.append(self.vocab[word])
                    else:
                        for char in word:
                            word_tokens.append(self.vocab.get(char, self.vocab[self.unk_token]))
                tokens.append(word_tokens)
                
        if isinstance(text, str):
            attention_mask = [1] * len(tokens)
            return {"input_ids": tokens, "attention_mask": attention_mask}
        else:
            attention_masks = [[1] * len(t) for t in tokens]
            return {"input_ids": tokens, "attention_mask": attention_masks}
        
    def decode(self, token_ids: Union[List[int], List[List[int]]], skip_special_tokens: bool = True) -> str:
        """Decode token ids to text"""
        # Create reverse vocab mapping
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        if isinstance(token_ids[0], list):
            # Batch decoding
            texts = []
            for ids in token_ids:
                text = []
                for id in ids:
                    token = id_to_token.get(id, self.unk_token)
                    if not skip_special_tokens or token not in [self.unk_token, self.pad_token, "<|mask|>"]:
                        text.append(token)
                texts.append(" ".join(text))
            return texts
        else:
            # Single sequence decoding
            text = []
            for id in token_ids:
                token = id_to_token.get(id, self.unk_token)
                if not skip_special_tokens or token not in [self.unk_token, self.pad_token, "<|mask|>"]:
                    text.append(token)
            return " ".join(text)