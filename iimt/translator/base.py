from ..common.base import Base
from ..common.logger import get_logger
from ..common.batch import get_batch

from typing import List

logger = get_logger('Translator')

class BaseTranslator(Base):
    def __init__(self, subbatch_size: int=10) -> None:
        super().__init__()
        self.subbatch_size = subbatch_size

    def forward(self, texts: List[str], **kargs) -> List[str]:
        texts_set = list(set(texts))
        translate_map = {}
        for batch in get_batch(texts_set, self.subbatch_size):
            batch_outputs = self.translate(batch, **kargs)
            translate_map.update({text: output for text, output in zip(batch, batch_outputs)})
        return [translate_map[text] for text in texts]
        
    def translate(self, texts: List[str], **kargs) -> List[str]:
        raise NotImplementedError()

