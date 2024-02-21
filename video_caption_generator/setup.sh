#wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh

../etc/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -u

export PATH=/root/miniconda3/bin:$PATH
conda create -n BLIPCaptions anaconda python=3.9
conda activate BLIPCaptions
ipython kernel install --user --name=BLIPCaptions

pip3 install salesforce-lavis