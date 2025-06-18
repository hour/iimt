from typing import List, Any
import numpy as np

from iimt.common.base import Base
from iimt.detector.base import BaseRecoDetector
from iimt.editor.base import BaseEditor
from iimt.translator.base import BaseTranslator
from iimt.common.media import Image
from iimt.common.batch import flatten, unflatten
from iimt.common.transform import transform
from iimt.common.insert import insert
from iimt.common.bbox import Bbox
from iimt.common.logger import get_logger

logger = get_logger('IIMT')

# [Not yet tested]
def filter(fn_skip_text, batch_texts, *args):
    skip_ids = [
        [j for j in range(len(batch_texts[i])) if fn_skip_text(batch_texts[i][j])]
        for i in range(len(batch_texts))
    ]

    def _filter(batch):
        output = []
        for i, eles in enumerate(batch):
            eles = [ele for j, ele in enumerate(eles) if j not in skip_ids[i]]
            if len(eles) > 0:
                output.append(eles)
        return output

    args = [_filter(arg) for arg in args]
    return _filter(batch_texts), *args

class IIMTPipeline(Base):
    def __init__(
        self,
        recodetector: BaseRecoDetector,
        editor: BaseEditor,
        translator: BaseTranslator,
        margin: float=0
    ) -> None:
        self.recodetector = recodetector
        self.editor = editor
        self.translator = translator
        self.margin = margin
        self.fix_bbox = Bbox.from_wh(width=900, height=300)

    def forward(self, imgs: List[Image], fn_skip_text=None) -> Any:
        def _transform(batch_bboxs):
            flat_transformed_imgs = []
            for bboxs, img in zip(batch_bboxs, imgs):
                for bbox in bboxs:
                    _bbox = bbox.add_margin(self.margin)
                    _img = transform(img, _bbox, self.fix_bbox)
                    flat_transformed_imgs.append(_img)
            return np.array(flat_transformed_imgs)

        def _edit(flat_translations, flat_transformed_imgs):
            if len(flat_translations) < 1:
                return []
            mask = np.char.strip(flat_translations) != ''
            flat_edited_imgs = np.full(flat_translations.shape, None, dtype=object)
            flat_edited_imgs[mask] = self.editor(
                flat_translations[mask],
                flat_transformed_imgs[mask]
            )
            return flat_edited_imgs

        batch_bboxs, batch_detection_scores, batch_texts, batch_recognition_scores = self.recodetector(imgs)
        logger.info('Done detection')

        if fn_skip_text is not None:
            batch_texts, batch_bboxs, batch_detection_scores, batch_recognition_scores = filter(
                fn_skip_text, batch_texts, batch_bboxs, batch_detection_scores, batch_recognition_scores
            )

        flat_translations = np.array(self.translator(flatten(batch_texts)))
        logger.info('Done translation')

        flat_transformed_imgs = _transform(batch_bboxs)
        flat_edited_imgs = _edit(flat_translations, flat_transformed_imgs)
        logger.info('Done edition')

        batch_edited_imgs = unflatten(flat_edited_imgs, batch_texts)

        output = []
        index = 0
        for img, bboxs, edited_imgs, texts in zip(imgs, batch_bboxs, batch_edited_imgs, batch_texts):
            for bbox, edited_img, text in zip(bboxs, edited_imgs, texts):
                if edited_img is None:
                    continue
                logger.info(f'{text} -> {flat_translations[index]}')
                index += 1
                img = insert(img, edited_img, bbox, margin=self.margin, copy=True)
            output.append(img)
        logger.info('All done')
        return output
