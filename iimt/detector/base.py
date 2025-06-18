from ..common.base import Base
from ..common.media import Image
from ..common.logger import get_logger
from ..common.bbox import Bbox
from ..common.transform import transform
from ..common.batch import get_batch, unflatten

from typing import List

logger = get_logger('Detector')

def pad_imags(imgs: List[Image]):
    max_width, max_height = 0, 0
    for img in imgs:
        if img.width > max_width:
            max_width = img.width
        if img.height > max_height:
            max_height = img.height
    return [
        img.pad(max_width-img.width, max_height-img.height) 
        for img in imgs
    ]

class BaseDetector(Base):
    def __init__(self, margin: float=0, padding: bool=True) -> None:
        self.margin = margin
        self.padding = padding
    
    def forward(self, imgs: List[Image]):
        assert(len(imgs) > 0)

        if self.padding:
            imgs = pad_imags(imgs)

        detected_bboxs, detected_scores = [], []
        for batch in get_batch(imgs):
            batch_detected_bboxs, batch_detected_scores = self.detect(batch)
            detected_bboxs += batch_detected_bboxs
            detected_scores += batch_detected_scores
        
        assert(len(detected_bboxs) == len(detected_scores) == len(imgs))
        # assert(isinstance(detected_bboxs[0], list) and isinstance(detected_bboxs[0][0], Bbox))
        assert(isinstance(batch_detected_bboxs[0], list))

        detected_bboxs = [
            [
                bbox.arrange().add_margin(self.margin) 
                for bbox in bboxs
            ]
            for bboxs in detected_bboxs
        ]
        return detected_bboxs, detected_scores

    def detect(self, imgs: List[Image]):
        raise NotImplementedError()

class BaseRecognizer(Base):
    def __init__(self, padding: bool=True) -> None:
        self.padding = padding
    
    def forward(self, imgs: List[Image]):
        assert(len(imgs) > 0)

        if self.padding:
            imgs = pad_imags(imgs)

        recognized_texts, recognized_scores = [], []
        for batch in get_batch(imgs):
            batch_recognized_texts, batch_recognized_scores = self.recognize(batch)
            recognized_texts += batch_recognized_texts
            recognized_scores += batch_recognized_scores
        return recognized_texts, recognized_scores

    def recognize(self, imgs: List[Image]):
        raise NotImplementedError()

class BaseRecoDetector(Base):
    def __init__(self, margin: float=0, padding: bool=True) -> None:
        self.margin = margin
        self.padding = padding
    
    def forward(self, imgs: List[Image]):
        assert(len(imgs) > 0)

        if self.padding:
            imgs = pad_imags(imgs)

        detected_bboxs, detected_scores, recognized_texts, recognized_scores = [], [], [], []
        for batch in get_batch(imgs):
            batch_detected_bboxs, batch_detected_scores, batch_recognized_texts, batch_recognized_scores = self.recodetect(batch)
            detected_bboxs += batch_detected_bboxs
            detected_scores += batch_detected_scores
            recognized_texts += batch_recognized_texts
            recognized_scores += batch_recognized_scores
        
        assert(len(detected_bboxs) == len(detected_scores) == len(imgs))
        # assert(isinstance(batch_detected_bboxs[0], list) and isinstance(batch_detected_bboxs[0][0], Bbox))
        assert(isinstance(detected_bboxs[0], list))

        detected_bboxs = [
            [
                bbox.arrange().add_margin(self.margin) 
                for bbox in bboxs
            ]
            for bboxs in detected_bboxs
        ]
        return detected_bboxs, detected_scores, recognized_texts, recognized_scores

    def recodetect(self, imgs: List[Image]):
        raise NotImplementedError()

class MergeRecoDetector(BaseRecoDetector):
    def __init__(
            self,
            recognizer: BaseRecognizer, 
            detector: BaseDetector, 
            transform: bool=False,
            subbatch_size: int=10,
        ) -> None:
        self.recognizer = recognizer
        self.detector = detector
        self.transform = transform
        self.subbatch_size = subbatch_size
        super().__init__(margin=self.detector.margin, padding=None)
    
    def crop(self, img: Image, bbox: Bbox) -> Image:
        rec_bbox = bbox.to_outbound_rectangle()
        if self.transform:
            return transform(img, bbox, rec_bbox)
        return img.crop(rec_bbox.topleft, rec_bbox.bottomright)

    def forward(self, imgs: List[Image]):
        batch_detected_bboxs, batch_detected_scores = self.detector(imgs)
        cropped_imgs = [
            self.crop(img, bbox)
            for img, bboxs in zip(imgs, batch_detected_bboxs)
            for bbox in bboxs
        ]

        batch_recognized_texts, batch_recognized_scores = self.recognizer(cropped_imgs)
        batch_recognized_texts = unflatten(batch_recognized_texts, batch_detected_bboxs)
        batch_recognized_scores = unflatten(batch_recognized_scores, batch_detected_bboxs)
        return batch_detected_bboxs, batch_detected_scores, batch_recognized_texts, batch_recognized_scores
