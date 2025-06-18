# In-Image Machine Translation
This is a pipeline to translate scene text from one language to another by leveraging publicly available models for text detection, recognition, and translation. We also provide a training script for SRNet to support cross-lingual image text replacement.

# Dependency
- Python==3.12.0
- CUDA==12.2.2

# Installation
```
conda create -n iimt
conda activate iimt
conda install python==3.12.0
conda install conda-forge::opencv
pip install -r requirements.txt
```

(Optional) The default translation model is NLLB. If you would like to use the "みんなの自動翻訳@TexTra" translation service, you need [trans](https://github.com/ideuchi/trans).
```
git clone https://github.com/ideuchi/trans

export HOME_TRANS=$(pwd)/trans
export TEXTRA_NAME=<your user_id>
export TEXTRA_KEY=<your api_key>
export TEXTRA_SECRET=<your api_secret>

# visit https://mt-auto-minhon-mlt.ucri.jgn-x.jp to get your free account
```

# Basic Usage
Build joint text detector and recognizer
```
from iimt.detector.doctr import DoctrRecoDetector
recodetector = DoctrRecoDetector(
    detector_name='fast_base',
    recognizer_name='crnn_vgg16_bn',
)
```

Build a translator
```
# NLLB model
from iimt.translator.nllb import NLLBTranslator
translator = NLLBTranslator(
    'facebook/nllb-200-distilled-600M',
    trg_lang='jpn_Jpan',
)

# or using the みんなの自動翻訳@TexTra service
from iimt.translator.textra import TexTraTranslator
translator = TexTraTranslator(
    src_lang='en',
    trg_lang='ja',
)
```

Build a text editor.\
You need to train your own model. Check [this](srnet/README.md) for SRNet training.
```
SRNET_MODEL_PATH=<your model path>
FONT_PATH=<your font path>

from iimt.editor.srnet import SRNetEditor
editor = SRNetEditor(
    model_path=SRNET_MODEL_PATH,
    font=FONT_PATH,
)
```

Build a IIMT pipeline
```
from iimt.pipeline import IIMTPipeline
IIMTPipeline(
    recodetector=recodetector,
    translator=translator,
    editor=editor,
    margin=30, # add margin to the detected bounding boxes before text editing
)
```

Translate images
```
from iimt.common.media import Image
imgs = [
    Image.load('data/image1.jpeg'),
    Image.load('data/image2.jpeg'),
]

out_imgs = pipeline(imgs)

for i, out_img in enumerate(out_imgs):
    out_img.save(f'data/out_image{i+1}.jpeg')
```

Translate video frame-by-frame
```
from iimt.common.media import Video
video = Video("data/video.mp4")

out_frames = pipeline(video.frames)

for frame_index, frame in enumerate(out_frames):
    video.replace(frame, frame_index)

video.save("data/out_video.mp4")
```

# Run translate.py
```
python translate.py <input_image_or_video_path> <output_image_or_video_path>

# with a configuration
python translate.py -c configs/textra.yaml <input_image_or_video_path> <output_image_or_video_path>
```

# Citation
```
@article{kaing2024towards,
  title={Towards Scene Text Translation for Complex Writing Systems},
  author={Kaing, Hour and Song, Haiyue and Ding, Chenchen and Mao, Jiannan and Tanaka, Hideki and Utiyama, Masao},
  journal={言語処理学会 第30回年次大会},
  year={2025}
}
```
