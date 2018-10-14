#!/bin/bash
if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <data-dir> <config-file> <results-dir> <weights-dir> <available-gpus> <options>"
    exit 1
fi
docker run -e "NVIDIA_VISIBLE_DEVICES=$5" --runtime=nvidia --rm \
    -v `pwd`/"$1":/data \
    -v `pwd`/`dirname $2`:/config \
    -v `pwd`/"$3":/results \
    -v `pwd`/"$4":/weights \
    yukw777/rsna-model python train.py /config/`basename $2` /data/stage_1_train_images /data/stage_1_train_labels.csv /results "${@:6}"
