from .base import BaseTranslator
from typing import List
import os

class TexTraTranslator(BaseTranslator):
    def __init__(
        self,
        src_lang: str='en',
        trg_lang: str='ja',
        subbatch_size: int=10,
    ) -> None:
        super().__init__(subbatch_size=subbatch_size)
        self.src_lang = src_lang
        self.trg_lang = trg_lang        
        self.cmd = os.environ.get("HOME_TRANS") + f"/trans text \"\" generalNT "
        
        assert(os.environ.get("HOME_TRANS") is not None), "export HOME_TRANS=<path to the `trans` home directory>"
        assert(os.environ.get("TEXTRA_NAME") is not None), "export TEXTRA_NAME=<your user_id>"
        assert(os.environ.get("TEXTRA_KEY") is not None), "export TEXTRA_KEY=<your api_key>"
        assert(os.environ.get("TEXTRA_SECRET") is not None), "export TEXTRA_SECRET=<your api_secret>"
        
    def to(self, device: str):
        pass

    def translate(self, texts: List[str], src_lang: str=None, trg_lang: str=None):
        if src_lang is None:
            src_lang = self.src_lang
        if trg_lang is None:
            trg_lang = self.trg_lang

        texts = "\n".join(texts)
        output = os.popen(
            f"echo -e \"{texts}\" | {self.cmd} {src_lang} {trg_lang}",
        ).read()
        return output.split('\n')
        
