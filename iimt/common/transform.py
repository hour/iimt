from .bbox import Bbox
from .media import Image

import cv2 as cv
import numpy as np

# forward: Spatial Transformer Network
# - https://github.com/sbillburg/CRNN-with-STN
# - https://github.com/kundank78/SpatialTransformer

# backward: inverse perspective transform
# - https://stackoverflow.com/questions/51827264/inverse-perspective-transform

def transform(img: Image, src_bbox: Bbox, trg_bbox: Bbox, width=None, height=None) -> Image:
    pts1 = np.float32(src_bbox.to_list())[[0,1,3,2]]
    pts2 = np.float32(trg_bbox.to_list())[[0,1,3,2]]

    if width is None or height is None:
        width, height = int(trg_bbox.max_x - trg_bbox.min_x), int(trg_bbox.max_y - trg_bbox.min_y)
        # xs, ys = list(zip(*trg_bbox.to_list()))
        # width, height = int(max(xs) - min(xs)), int(max(ys) - min(ys))

    M = cv.getPerspectiveTransform(pts1,pts2)
    dst = cv.warpPerspective(img.image.astype('uint8'),M,(width,height))
    return Image(image=dst.astype('uint8'), channel_first=img.channel_first)
