from .base import BaseTranslator

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List

class NLLBTranslator(BaseTranslator):
    NLLB_MODELS = [
        "facebook/nllb-200-distilled-600M"
    ]
    
    def __init__(self, name, trg_lang: str=None, subbatch_size: int=10) -> None:
        super().__init__(subbatch_size=subbatch_size)
        assert(name in self.NLLB_MODELS)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name)
        self.device = 'cpu'
        self.trg_lang = trg_lang

    def to(self, device: str):
        self.device = device
        self.model.to(self.device)

    def translate(self, texts: List[str], trg_lang: str=None):
        if trg_lang is None:
            trg_lang = self.trg_lang
        assert(trg_lang is not None), 'Specify `trg_lang`.'
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        translated_tokens = self.model.generate(
            **inputs, forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(trg_lang), max_length=30
        )
        outputs = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return outputs
        
