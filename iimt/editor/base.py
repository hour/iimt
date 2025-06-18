from ..common.base import Base
from ..common.media import Image
from ..common.logger import get_logger

from typing import List
from tqdm import tqdm

logger = get_logger('Editor')

class BaseEditor(Base):
    def __init__(self, subbatch_size=10) -> None:
        self.subbatch_size = subbatch_size
    
    def forward(self, texts: List[str], imgs: List[Image]) -> List[Image]:
        def _get_batch():
            out_texts, out_imgs = [], []
            for text, img in tqdm(zip(texts, imgs), total=len(texts)):
                if len(out_texts) > self.subbatch_size:
                    yield out_texts, out_imgs
                    out_texts, out_imgs = [], []
                out_texts.append(text)
                out_imgs.append(img)
            if len(out_texts) > 0:
                yield out_texts, out_imgs
        
        output = []
        for batch_texts, batch_imgs in _get_batch():
            output += self.edit(batch_texts, batch_imgs)
        return output

    def edit(self, batch_texts: List[str], batch_imgs: List[Image]) -> List[Image]:
        raise NotImplementedError()

