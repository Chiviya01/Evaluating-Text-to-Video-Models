apt-get update
apt-get -y install nano
apt install libgl1-mesa-glx
apt-get install libglib2.0-0 -y
apt install -y libsm6 libxext6
apt-get install -y libxrender-dev
apt install ffmpeg
apt install nvidia-cuda-toolkit
apt-get install wget

#wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

etc/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -u

export PATH=/root/miniconda3/bin:$PATH

. ~/miniconda3/bin/activate
conda init bash