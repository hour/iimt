# Requiring Data
```
.
|-- japanese.txt
|-- fonts/
|   |-- english_ttf
|   |   |-- arial.ttf
|   |-- japanese_ttf
|   |   |-- HinaMincho-Regular.ttf
|-- imnames.cp
|-- bg_img/
```

# Japanese words
Prepare a list of Japanese words and put them (line by line) in a `japanese.txt` file.
Check [this](https://github.com/hingston/japanese).

# Fonts
For standard fonts, download and put them in
- fonts/english_ttf/arial.ttf
- fonts/japanese_ttf/HinaMincho-Regular.ttf ([download](https://fonts.google.com/specimen/Hina+Mincho))

Besides the standard fonts, you can add more fonts for training. For examples,
- fonts/english_ttf/OpenSans-Regular.ttf ([download](https://fonts.google.com/specimen/Open+Sans))
- fonts/japanese_ttf/NotoSerifJP-VariableFont_wght.ttf ([download](https://fonts.google.com/noto/specimen/Noto+Serif+JP))

# Background Image
Prepare background images as follows
- bg_img/* : a dictory containing background images
- imnames.cp : list of file names in bg_img/*

Check [this](https://github.com/ankush-me/SynthText) to obtain the above background images.
