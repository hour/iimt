from .base import BaseDetector, BaseRecognizer, BaseRecoDetector
from ..common.media import Image
from ..common.bbox import Bbox, Coordinate
from ..common.logger import get_logger

from doctr.io.elements import Page

from typing import List
import torch
import numpy as np

logger = get_logger('DoctrDetector')

# methods
# - FAST: https://github.com/czczup/FAST?tab=readme-ov-file
# - EAST: https://github.com/argman/EAST
# - tensorflow_PSENet: https://github.com/liuheng92/tensorflow_PSENet

class DoctrDetector(BaseDetector):
    # check doctr for model name
    # e.g., db_resnet50, fast_base
    def __init__(self, name, margin=0, padding=True, assume_straight_pages=False) -> None:
        super().__init__(margin=margin, padding=padding)
        from doctr.models import detection

        # self.model = detection.fast_tiny(pretrained=True).eval()
        self.model = detection.detection_predictor(
            name,
            pretrained=True,
            preserve_aspect_ratio=False,
            symmetric_pad=True,
            assume_straight_pages=assume_straight_pages, # [TODO] this should be false because of scense image
        )

    def to(self, device: str):
        self.model.to(device)

    def detect(self, imgs: List[Image]):
        def _format_output(words):
            bboxs, scores = [], []
            for word in words:
                assert(word.shape == (5,2))
                bboxs.append(
                    Bbox(
                        topleft=Coordinate(*word[0]),
                        topright=Coordinate(*word[1]),
                        bottomright=Coordinate(*word[2]),
                        bottomleft=Coordinate(*word[3])
                    )
                )
                scores.append(word[4][1])
            return bboxs, scores                
        
        imgs_tensor = torch.stack([img.to_channel_first().to_tensor() for img in imgs])
        _, _, height, width = imgs_tensor.shape

        outs = self.model(imgs_tensor)

        # Convert to absolute coordinates
        for batch_id in range(len(outs)):
            outs[batch_id]['words'][:, :-1, 0] *= width
            outs[batch_id]['words'][:, :-1, 1] *= height
        
        batch_bboxs, batch_scores = [], []
        for out in outs:
            bboxs, scores = _format_output(out['words'])
            batch_bboxs.append(bboxs)
            batch_scores.append(scores)
        return batch_bboxs, batch_scores

class DoctrRecognizer(BaseRecognizer):
    # check doctr for model name
    # e.g., db_resnet50, fast_base
    def __init__(self, name, padding=True) -> None:
        super().__init__(padding=padding)
        from doctr.models import recognition
        self.model = recognition.recognition_predictor(
            name,
            pretrained=True,
            symmetric_pad=True,
        )

    def to(self, device: str):
        self.model.to(device)

    def recognize(self, imgs: List[Image]):
        imgs_tensor = torch.stack([img.to_channel_first().to_tensor() for img in imgs])
        return list(zip(*self.model(imgs_tensor)))

class DoctrRecoDetector(BaseRecoDetector):
    def __init__(
            self,
            detector_name: str,
            recognizer_name: str,
            min_detection_score: float=0, 
            min_recognition_score: float=0, 
            margin=0,
            padding=False,
            assume_straight_pages=False,
        ) -> None:
        super().__init__(margin=margin, padding=padding)
        
        from doctr.models import ocr_predictor
        args = {
            'preserve_aspect_ratio': False,
            'symmetric_pad': True,
            'assume_straight_pages': assume_straight_pages,
        }
        self.model = ocr_predictor(
            det_arch=detector_name,
            reco_arch=recognizer_name,
            pretrained=True,
            **args
        )
        self.min_detection_score = min_detection_score
        self.min_recognition_score = min_recognition_score

    def to(self, device: str):
        self.model.to(device)

    def extract_page(self, page: Page):
        try:
            words, bboxs, recognition_scores, detection_scores = list(zip(*[
                (word.value, Bbox.from_np(np.array(word.geometry)), word.confidence, word.objectness_score) 
                for block in page.blocks for line in block.lines for word in line.words
                if word.confidence >= self.min_recognition_score and word.objectness_score >= self.min_detection_score
            ]))
            return words, bboxs, recognition_scores, detection_scores
        except:
            return [], [], [], []

    def recodetect(self, imgs: List[Image]):
        output = self.model([img.image for img in imgs])

        batch_words, batch_bboxs, batch_recognition_scores, batch_detection_scores = [], [], [], []
        for img, page in zip(imgs, output.pages):
            words, bboxs, recognition_scores, detection_scores = self.extract_page(page)
            assert(len(words) == len(bboxs) == len(recognition_scores) == len(detection_scores))
            bboxs = [bbox.scale(img.width, img.height) for bbox in bboxs]
            batch_words.append(words)
            batch_bboxs.append(bboxs)
            batch_recognition_scores.append(recognition_scores)
            batch_detection_scores.append(detection_scores)
        assert(len(batch_words) == len(batch_bboxs) == len(batch_recognition_scores) == len(batch_detection_scores))
        return batch_bboxs, batch_detection_scores, batch_words, batch_recognition_scores
