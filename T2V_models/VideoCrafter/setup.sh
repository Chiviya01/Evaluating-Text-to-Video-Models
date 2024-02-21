apt update
apt install software-properties-common
apt install ffmpeg &> /dev/null 

export PATH=/root/miniconda3/bin:$PATH
conda create -n VideoCrafter anaconda python=3.8
conda activate VideoCrafter
ipython kernel install --user --name=VideoCrafter

dpkg --remove --force-remove-reinstreq python3-pip python3-setuptools python3-wheel
apt-get install python3-pip

git clone https://github.com/VideoCrafter/VideoCrafter &> /dev/null
cd VideoCrafter 
export PYTHONPATH=/notebooks/T2V_models/VideoCrafter/VideoCrafter/VideoCrafter:$PYTHONPATH

#python3.8 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.12.1 -f https://download.pytorch.org/whl/torch_stable.html
 
python3.8 -m pip install pytorch-lightning==1.8.3 omegaconf==2.1.1 einops==0.3.0 transformers==4.25.1
python3.8 -m pip install opencv-python==4.1.2.30 imageio==2.9.0 imageio-ffmpeg==0.4.2
python3.8 -m pip install av moviepy
python3.8 -m pip install -e .