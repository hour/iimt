#!/bin/bash

# for i in `seq 1 10`;
# do
#     num=$(printf "%03d" $i)
#     cp ../SRNet-Datagen/outputs/i_s/*0${i}.png labels/${num}_i_s.png || exit 0
#     cp ../SRNet-Datagen/outputs/i_t/*0${i}.png labels/${num}_i_t.png || exit 0
# done

counter=1
for file in `ls ../SRNet-Datagen/outputs/i_s`;
do
    echo $file
    extension="${file##*.}"

    cp ../SRNet-Datagen/outputs/i_s/$file labels/${counter}_i_s.${extension} || exit 0
    cp ../SRNet-Datagen/outputs/i_t/$file labels/${counter}_i_t.${extension} || exit 0

    ((counter=counter+1))
    if [ $counter -gt 10 ];
    then
        break
    fi
done