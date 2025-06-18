# Dependency
- CUDA==12.6.3
- Python==3.9.0

# Installation
- conda create -n srnet
- conda activate srnet
- conda install python==3.9.0
- pip install -r requirements.txt

# Prepare Data
Check [this](./SRNet-Datagen/README.md).

# Training
```
python train.py
```
Various model checkpoints will be saved in `models/`.

# Finetuning
Make sure a pre-trained chekcpoint named `pretrained/trained_final_5M_.model` exists before runing `python train.py`. Check [this](https://github.com/lksshw/SRNet/tree/master?tab=readme-ov-file#pre-trained-weights).

# Acknowledgments
This repository is built on top of [SRNet](https://github.com/lksshw/SRNet), [SRNet-Datagen](https://github.com/youdao-ai/SRNet-Datagen), and [SynthText](https://github.com/ankush-me/SynthText).