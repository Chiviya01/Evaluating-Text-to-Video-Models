# Video Creation using VideoFusion

This directory is used to generator videos using the VideoFusion algorithm.
We provide the ./setup.sh file to setup the enviroment need to run the Python notebook Model.ipynb and the text file extra_requirements.txt include the Python libraries need to run the code.

### Directory Structure
```
./
├── Model.ipynb (Python model used to create videos)
├── setup.sh (Optional setup to run Python notebook)
└── extra_requirements.txt (Python libraries need to run code)
```

---
## Enviroment Setup
You can skip the setup made through the ./setup.sh script but we suggest to create a Python virtual environment to run the code.
We also suggestion using conda to mange the enviroment, using the following commands.

```

$ conda create -n VideoFusion anaconda python=3.9
$ conda activate VideoFusion
$ ipython kernel install --user --name=VideoFusion

$ pip install torch==2.0.0 git+https://github.com/huggingface/diffusers transformers accelerate decorator imageio[ffmpeg]
$ pip install -r extra_requirements.txt
```

---
## Running the Notebook
Before running the notebook make sure that you have selected the correct kernel "VideoFusion"
In the notebook you can generate new videos and also view the generated videos.

---
## Output
The output of the notebook is saved to the directory "../../generated_videos/VideoFusion/"
Each video is named after the caption used to generate it.