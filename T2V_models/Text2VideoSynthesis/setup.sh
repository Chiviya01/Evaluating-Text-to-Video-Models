#wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

../../etc/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -u
export PATH=/root/miniconda3/bin:$PATH
conda create -n Text2VideoSynthesis anaconda python=3.9
conda activate Text2VideoSynthesis
ipython kernel install --user --name=Text2VideoSynthesis

pip install modelscope==1.4.2
pip install open_clip_torch
pip install pytorch-lightning