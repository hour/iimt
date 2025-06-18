from iimt.common.logger import get_logger
import mimetypes

from dataclasses import dataclass
import yaml

logger = get_logger('IIMT')

@dataclass
class Config:
    detector_name: str="fast_base"
    recognizer_name: str='crnn_vgg16_bn'
    editor_model_path: str='srnet/models/train_step-500000.model'
    editor_font_path: str='srnet/SRNet-Datagen/data/fonts/japanese_ttf/HinaMincho-Regular.ttf'
    editor_bbox_margin: float=30
    translator_name: str='nllb'
    translator_model_name: str='facebook/nllb-200-distilled-600M'
    translator_trg_lang: str='jpn_Jpan'
    translator_textra_user_id: str=None,
    translator_textra_api_key: str=None,
    translator_textra_api_secret: str=None,
    common_gpu: bool=True,
    
    @classmethod
    def load_yaml(cls, path: str):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        
        flat_config = {}
        for module_name, properties in config.items():
            for key, value in properties.items():
                flat_config[f'{module_name}_{key}'] = value
        return cls(**flat_config)

def build_pipeline(configs):
    from iimt.pipeline import IIMTPipeline
    from iimt.detector.doctr import DoctrRecoDetector
    from iimt.editor.srnet import SRNetEditor

    recodetector = DoctrRecoDetector(
        detector_name=configs.detector_name,
        recognizer_name=configs.recognizer_name,
    )

    editor = SRNetEditor(
        model_path=configs.editor_model_path,
        font=configs.editor_font_path,
        padding_lr=0,
        padding_tb=0,
    )

    if configs.translator_name == 'nllb':
        from iimt.translator.nllb import NLLBTranslator
        translator = NLLBTranslator(
            configs.translator_model_name,
            trg_lang=configs.translator_trg_lang
        )
    elif configs.translator_name == 'textra':
        from iimt.translator.textra import TexTraTranslator
        translator = TexTraTranslator(
            user_id=configs.translator_textra_user_id,
            api_key=configs.translator_textra_api_key,
            api_secret=configs.translator_textra_api_secret,
            src_lang='en',
            trg_lang=configs.translator_trg_lang
        )

    if configs.common_gpu:
        recodetector.to('cuda')
        editor.to('cuda')
        translator.to('cuda')

    return IIMTPipeline(
        recodetector=recodetector,
        editor=editor,
        translator=translator,
        margin=configs.editor_bbox_margin,
    )

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('output')
    parser.add_argument('--config', '-c')
    args = parser.parse_args()

    filetype = mimetypes.guess_type(args.input)[0]
    
    if args.config is not None:
        config = yaml.load(args.config)
    else:
        config = Config()
    
    pipeline = build_pipeline(config)

    if 'video' in filetype:
        from iimt.common.media import Video
        video = Video(args.input)
        logger.info(f'Vidoe frames: {len(video)}')
        for i, img in enumerate(pipeline(video.frames)):
            video.replace(img, i)
        video.save(args.output)
    
    elif 'image' in filetype:
        from iimt.common.media import Image
        img = Image.load(args.input)
        pipeline([img])[0].save(args.output)

    else:
        raise ValueError(f'Unknown type {filetype}')