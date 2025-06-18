from .bbox import Bbox
from .media import Image
from .transform import transform

from PIL import Image as PILImage
from PIL import ImageDraw
import numpy as np
import math

def get_mask(img, bbox):
    maskIm = PILImage.new('L', (img.width, img.height), 0)
    ImageDraw.Draw(maskIm).polygon(bbox.to_list(), outline=0, fill=1)
    return np.array(maskIm) != 0

def insert(main_img: Image, insert_img: Image, bbox: Bbox, margin=0, copy=False) -> Image:
    if copy:
        main_img = main_img.copy()
        
    mask = get_mask(main_img, bbox)

    insert_img = insert_img.resize(
        math.floor(bbox.max_x - bbox.min_x + margin + margin),
        math.floor(bbox.max_y - bbox.min_y + margin + margin)
    )
    org_bbox = Bbox.from_wh(insert_img.width, insert_img.height).add_margin(-margin)
    trg_bbox = bbox
    insert_img = transform(insert_img, org_bbox, trg_bbox, main_img.width, main_img.height)
    insert_mask = get_mask(insert_img, trg_bbox)
    main_img.image[mask] = insert_img.image[insert_mask]

    return main_img

# def insert(main_img: Image, insert_img: Image, bbox: Bbox, copy=True) -> Image:
#     if copy:
#         main_img = main_img.copy()
        
#     mask = get_mask(main_img, bbox)
#     org_bbox = Bbox.from_wh(insert_img.width, insert_img.height)
#     trg_bbox = bbox
#     insert_img = transform(insert_img, org_bbox, trg_bbox, main_img.width, main_img.height)
#     insert_mask = get_mask(insert_img, trg_bbox)
#     main_img.image[mask] = insert_img.image[insert_mask]
#     return main_img