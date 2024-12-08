# data/processors/tokenizer.py
# data/processors/text_cleaner.py
from config.config import Config
from transformers import AutoTokenizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from typing import List, Dict, Union


class FinanceTokenizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        self.max_length = Config.MAX_LENGTH

    def tokenize(self, texts: List[str]) -> Dict[str, List]:
        """Tokenize texts for transformer models."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask']
        }

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
