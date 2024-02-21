# Video Creation using ModelScope Text-to-Video Synthesis

This directory is used to generator videos using the Text-to-Video Synthesis algorithm.
We provide the ./setup.sh file to setup the enviroment need to run the Python notebook Model.ipynb.

### Directory Structure
```
./
├── Model.ipynb (Python model used to create videos)
└── setup.sh (Optional setup to run Python notebook)
```

---
## Enviroment Setup
You can skip the setup made through the ./setup.sh script but we suggest to create a Python virtual environment to run the code.
We also suggestion using conda to mange the enviroment, using the following commands.

```
conda create -n Text2VideoSynthesis anaconda python=3.9
conda activate Text2VideoSynthesis
ipython kernel install --user --name=Text2VideoSynthesis

pip install modelscope==1.4.2
pip install open_clip_torch
pip install pytorch-lightning
```

---
## Running the Notebook
Before running the notebook make sure that you have selected the correct kernel "Text2VideoSynthesis"
In the notebook you can generate new videos and also view the generated videos.

---
## Output
The output of the notebook is saved to the directory "../../generated_videos/Text2VideoSynthesis/"
Each video is named after the caption used to generate it.