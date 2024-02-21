#wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# /src/notebooks/etc/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -u

export PATH=/root/miniconda3/bin:$PATH
conda init bash
conda create -n Video_Naturalness anaconda python=3.9
conda activate Video_Naturalness
ipython kernel install --user --name=Video_Naturalness

apt-get update
apt-get install libgl1-mesa-glx
apt-get install libglib2.0-0
apt-get install ffmpeg

#installing all necessary packages
pip3 install scikit-video
pip3 install sentence_transformers
pip3 install pyiqa
pip install xgboost

#!/bin/bash
