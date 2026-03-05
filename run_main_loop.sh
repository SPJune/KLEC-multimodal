#!/bin/bash

#upstream_list=("hubert" "wav2vec2" "decoar2" "wavlm" "data2vec" "tera" "apc")
#set=$(seq 6 16)
#target_list=("hubert" "wav2vec2" "decoar2" "wavlm" "data2vec")
target_list=("hubert")
for target in "${target_list[@]}"
do
    #for i in $set
    for i in 0 1 2 3 4 5 7 8 9 10 11 12
    do
        python main.py exp_name=240807$target$i feature=$target feature.layer=$i
	done

done

