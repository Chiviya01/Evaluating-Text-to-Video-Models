# Video Creation using Open-Source T2V Models
This directory is used to generator videos using the various Text-to-Video algorithms.
Each directory includes a README.md file with further instructions and we provide the ./setup.sh file to setup the enviroment need to run the models.

### Directory Structure
```
./
├── Model (Python model used to create videos)
└── setup.sh (Optional setup to run Python notebook)
```

---
## Enviroment Setup
You can skip the setup made through the ./setup.sh script but we suggest to create a Python virtual environment to run the code.
We also suggestion using conda to mange the enviroment.

Make sure to run the ../setup.sh script before running any of the models.

---
## Running the Model
Before running the notebook make sure that you have selected the correct Python enviroment.

---
## Output
The output of the notebook is saved to the directory "../../generated_videos/$MODEL_TYPE/"
Each video is named after the caption used to generate it.

---
## Other Models. These models have available google collab to generate videos.
Aphantasia https://github.com/eps696/aphantasia.
Tune-a-Video https://github.com/showlab/Tune-A-Video.  
Text2Video-Zero https://github.com/Picsart-AI-Research/Text2Video-Zero. 