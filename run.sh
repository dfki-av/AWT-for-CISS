#!/bin/bash

pip install tensorboardX==1.8
pip install matplotlib
pip install numpy
pip install inplace-abn
pip install captum

bash scripts/voc/mib_voc_15-1.sh
