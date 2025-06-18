"""
Some configurations.
Copyright (c) 2019 Netease Youdao Information Technology Co.,Ltd.
Licensed under the GPL License 
Written by Yu Qian
"""
import numpy as np

# font
font_size = [25, 60]
# font_size = [59, 60]
underline_rate = 0.01
strong_rate = 0.07
oblique_rate = 0.02
src_font_dir = '../../data/fonts/english_ttf/'
src_standard_font_path = '../../data/fonts/english_ttf/arial.ttf'
trg_font_dir = '../../data/fonts/japanese_ttf/'
trg_standard_font_path = '../../data/fonts/japanese_ttf/HinaMincho-Regular.ttf'

# text
src_text_filepath = 'data/texts.txt'
trg_text_filepath = '../../data/japanese.txt'
capitalize_rate = 0.1
uppercase_rate = 0.04

# background
bg_filepath = '../../data/imnames.cp'
temp_bg_path = '../../data/bg_img/'

## background augment
brightness_rate = 0.8
brightness_min = 0.7
brightness_max = 1.5
color_rate = 0.8
color_min =0.7
color_max = 1.3
contrast_rate = 0.8
contrast_min = 0.7
contrast_max = 1.3

# curve
is_curve_rate = 0.05
curve_rate_param = [0.1, 0] # scale, shift for np.random.randn()

# perspective
rotate_param = [1, 0] # scale, shift for np.random.randn()
zoom_param = [0.1, 1] # scale, shift for np.random.randn()
shear_param = [2, 0] # scale, shift for np.random.randn()
perspect_param = [0.0005, 0] # scale, shift for np.random.randn()

# render

## surf augment
elastic_rate = 0.001
elastic_grid_size = 4
elastic_magnitude = 2

## colorize
padding_ud = [0, 10]
padding_lr = [0, 20]
is_border_rate = 0.02
is_shadow_rate = 0.02
shadow_angle_degree = [1, 3, 5, 7] # shift for shadow_angle_param
shadow_angle_param = [0.5, None] # scale, shift for np.random.randn()
shadow_shift_param = np.array([[0, 1, 3], [2, 7, 15]], dtype = np.float32) # scale, shift for np.random.randn()
shadow_opacity_param = [0.1, 0.5] # shift for shadow_angle_param
color_filepath = 'data/colors_new.cp'
use_random_color_rate = 0.5
