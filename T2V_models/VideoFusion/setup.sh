#!/bin/sh

export PATH=/root/miniconda3/bin:$PATH
conda create -n VideoFusion anaconda python=3.9
conda activate VideoFusion
ipython kernel install --user --name=VideoFusion

pip install torch==2.0.0 git+https://github.com/huggingface/diffusers transformers accelerate decorator imageio[ffmpeg]
pip install -r requirements.txt