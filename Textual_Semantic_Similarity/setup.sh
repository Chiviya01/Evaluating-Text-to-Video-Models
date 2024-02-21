#wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

# /src/notebooks/etc/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -u

export PATH=/root/miniconda3/bin:$PATH
conda init bash
conda create -n Text_Similarity anaconda python=3.9
conda activate Text_Similarity
ipython kernel install --user --name=Text_Similarity

apt-get update
apt-get install -y libgl1-mesa-glx
apt-get install libglib2.0-0

#installing all necessary packages
pip3 install transformers
pip install spacy
python3 -m spacy download en_core_web_sm
pip install nltk

#!/bin/bash