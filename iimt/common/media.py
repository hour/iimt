from .bbox import Bbox
from .logger import get_logger

import numpy as np
from dataclasses import dataclass
import torch, cv2
from tqdm import tqdm
from typing import Self

from PIL import Image as PILImage
from PIL import ImageFont
from PIL import ImageDraw

logger = get_logger('Media')

# library:
# - pytorchvideo

# references:
# - https://huggingface.co/docs/transformers/en/tasks/video_classification
# - https://pytorchvideo.org/docs/tutorial_torchhub_detection_inference
# - https://mindee.github.io/doctr/modules/models.html#

@dataclass
class Image:
    image: np.ndarray
    channel_first: bool = False

    @property
    def size(self):
        return self.image.shape

    @property
    def width(self):
        if self.channel_first:
            return self.size[2]
        return self.size[1]

    @property
    def height(self):
        if self.channel_first:
            return self.size[1]
        return self.size[0]
    
    @property
    def pil(self):
        return PILImage.fromarray(self.image.astype('uint8'), 'RGB')

    @classmethod
    def from_pil(cls, img_pil: PILImage):
        return cls(image=np.asarray(img_pil.convert('RGB')))

    def crop(self, left: float, top: float, right: float, bottom: float):
        return Image.from_pil(self.pil.crop((left, top, right, bottom)))

    def add_margin(self, left: float, top: float, right: float, bottom: float):
        return Image.from_pil(self.pil.crop((
            -left, -top, self.width + right, self.height + bottom
        )))

    def paste(self, image: Self, bbox: Bbox):
        img = self.pil
        img.paste(image.pil, (int(bbox.topleft.x), int(bbox.topleft.y)))
        return Image(
            image=np.array(img),
            channel_first=self.channel_first
        )

    def resize(self, width, height):
        if width == self.width and height == self.height:
            return self
        
        img = self.pil
        img = img.resize((width, height))
        return Image(
            image=np.asarray(img),
            channel_first=self.channel_first
        )

    def to_channel_first(self):
        if self.channel_first:
            return self
        return Image(
            image=np.stack((self.image[:,:,0], self.image[:,:,1], self.image[:,:,2]), axis=0),
            channel_first=True
        )

    def to_channel_last(self):
        if not self.channel_first:
            return self
        return Image(
            image=np.stack((self.image[0], self.image[1], self.image[2]), axis=2),
            channel_first=False
        )

    def to_tensor(self, dtype=torch.uint8):
        return torch.from_numpy(self.image).type(dtype)
    
    def pad(self, right, bottom, left=0, top=0, mode='constant', constant_values=0):
        if self.channel_first:
            image = np.pad(self.image, ((0, 0), (top, bottom), (left, right)), mode=mode, constant_values=constant_values)
        else:
            image = np.pad(self.image, ((top, bottom), (left, right), (0, 0)), mode=mode, constant_values=constant_values)

        return Image(
            image=image,
            channel_first=self.channel_first
        )
    
    def save(self, path):
        self.pil.save(path)

    @classmethod
    def load(cls, path):
        img = PILImage.open(path).convert('RGB')
        return cls(image=np.asarray(img))
    
    def copy(self):
        return Image(
            image=np.copy(self.image),
            channel_first=self.channel_first
        )
    
class Video:
    def __init__(self, path):
        vidcap = cv2.VideoCapture(path)
        self.fourcc = int(vidcap.get(cv2.CAP_PROP_FOURCC))
        # self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.fps = vidcap.get(cv2.CAP_PROP_FPS)
        self.frames = []

        pbar = tqdm()
        while True:
            success, np_img = vidcap.read()
            if not success:
                break
            self.frames.append(Image(image=np_img))
            pbar.update(1)
        pbar.close()

        vidcap.release()

    def __getitem__(self, frame_index) -> Image:
        return self.frames[frame_index]

    def __len__(self):
        return len(self.frames)

    @property
    def size(self):
        return self.frames[0].size

    @property
    def width(self):
        return self.frames[0].width

    @property
    def height(self):
        return self.frames[0].height

    # def __call__(self, frame_index) -> Image:
    #     return self.frames[frame_index]

    def replace(self, img: Image, frame_index):
        self.frames[frame_index] = img

    def save(self, path):
        video_writer = cv2.VideoWriter(path, 
            self.fourcc, self.fps, 
            (self.width, self.height)
        )

        for idx, img in enumerate(self):
            video_writer.write(img.image.astype(np.uint8))

        video_writer.release()

class TextImage:
    def __init__(
            self,
            text: str, 
            font: str, 
            width: int, 
            height: int, 
            min_font_size: int, 
            padding_lr: float=0,
            padding_tb: float=0,
            background='grey',
            fill='black',
        ):
        self.text = text
        self.width = width
        self.height = height
        self.background = background
        self.fill = fill
        self.padding_lr = padding_lr
        self.padding_tb = padding_tb
        self.font, self.center_left, self.center_top = self.init_font(
            font, min_font_size
        ) # name or path

    def init_font(self, font, min_font_size):
        def _width_height(size):
            left, top, right, bottom = ImageFont.truetype(f'{font}', size).getbbox(self.text)
            width = right + left
            height = bottom + top
            return width, height

        max_font_size = self.height
        offset_width, offset_height = self.padding_lr*2, self.padding_tb*2
        size = None
        for _size in range(max_font_size, min_font_size, -1):
            width, height = _width_height(_size)
            if width <= self.width + offset_width and height <= self.height + offset_height:
                size = _size
                break
            # print(_size, width, height)
        if size is None:
            size = min_font_size
            logger.warn(
                f'No space for text with h={self.height}, w={self.width}, '
                f'pad_lr={self.padding_lr}, pad_tb={self.padding_tb}. '
                f'Font size set to {min_font_size}'
            )
        width, height = _width_height(size)
        center_left = (self.width - width) / 2
        center_top = (self.height - height) / 2
        return ImageFont.truetype(f'{font}', size), center_left, center_top

    def draw(self) -> PILImage:
        img = PILImage.new('RGB', (self.width, self.height), self.background) # black background
        draw = ImageDraw.Draw(img)
        draw.text(
            (self.center_left, self.center_top), # left spacing and vertically middle
            self.text,
            fill=self.fill,
            font=self.font,
        )
        
        assert(not (np.asarray(img)==self.background).all())
        return img