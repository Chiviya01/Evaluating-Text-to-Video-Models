# Video Information Intraction using GDINO

This directory is used to extract the Average Precision of a the frames of a generated video based on the input prompt used to generate the video and the generated BLIP captions of the frame.

We provide the ./setup.sh file to setup the enviroment need to run the models.

### Directory Structure
```
./
├── Model.py (Python model used to generate the bounding boxes and AP)
├── AveragePrecision.py (Python code that calculates the AP of the bounding boxes found)
└── setup.sh (Optional setup to run Python notebook)
```

---
## Enviroment Setup
You can skip the setup made through the ./setup.sh script but we suggest to create a Python virtual environment to run the code.
We also suggestion using conda to mange the enviroment.

```
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
```
Make sure to run the ../setup.sh script before running any of the models.


---
## Running the Model
Before running the notebook make sure that you have selected the correct Python enviroment.

You can run the script by running the script "Model.py". This file uses the AveragePrecision.py script to calculate the AP for each video generated.

---
## Output
The output of the notebook is saved to the directory "../../output/bounding_boxes.json" and "../../output/ap_values.json"
The json files are stuctured such that the first key is the model typ and then the video name while the bounding_boxes.json includes values for each frame.