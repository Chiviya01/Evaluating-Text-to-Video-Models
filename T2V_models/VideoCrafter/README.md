# Video Creation using VideoCrafter

This directory is used to generator videos using the VideoCrafter algorithm.
We provide the ./setup.sh file to setup the enviroment need to run the Python notebook Model.ipynb.

### Directory Structure
```
./
├── Model.ipynb (Python model used to create videos)
├── setup.sh (Optional setup to run Python notebook)
└── VideoCrafter/ (Code used to run model)
```

---
## Enviroment Setup
You can skip the setup made through the ./setup.sh script but we suggest to create a Python virtual environment to run the code.
We also suggestion using conda to mange the enviroment, using the following commands.

```
conda create -n VideoCrafter anaconda python=3.8
conda activate VideoCrafter
ipython kernel install --user --name=VideoCrafter

dpkg --remove --force-remove-reinstreq python3-pip python3-setuptools python3-wheel
apt-get install python3-pip

git clone https://github.com/VideoCrafter/VideoCrafter &> /dev/null
cd VideoCrafter 
export PYTHONPATH=/T2V_models/VideoCrafter/VideoCrafter/VideoCrafter:$PYTHONPATH 

python3.8 -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
 
python3.8 -m pip install pytorch-lightning==1.8.3 omegaconf==2.1.1 einops==0.3.0 transformers==4.25.1
python3.8 -m pip install opencv-python==4.1.2.30 imageio==2.9.0 imageio-ffmpeg==0.4.2
python3.8 -m pip install av moviepy
python3.8 -m pip install -e .
```

---
## Running the Notebook
Before running the notebook make sure that you have selected the correct kernel "VideoCrafter"
In the notebook you can generate new videos and also view the generated videos.

---
## Output
The output of the notebook is saved to the directory "../../generated_videos/VideoCrafter/"
Each video is named after the caption used to generate it.