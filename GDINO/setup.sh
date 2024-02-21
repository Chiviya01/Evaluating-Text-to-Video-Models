export PATH=/root/miniconda3/bin:$PATH

. ~/miniconda3/bin/activate
conda init bash

conda create -n GDINO anaconda python=3.7
conda activate GDINO
ipython kernel install --user --name=GDINO

git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -r requirements.txt
pip install -q -e .
pip install -q roboflow

pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html

cd ..
mkdir weights/
cd weights/

wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

cd ..
mkdir data/
cd data/