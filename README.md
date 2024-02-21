# Evaluating Text-to-Video Models

This project represents a convergence of machine learning, video and image processing, and natural language processing (NLP) in pursuit of a novel and innovative quality evaluation metric for video output from text-to-video models. The project combines a custom-built image naturalness and human interpretability classifier with advanced text similarity techniques to produce more precise and dependable performance scores. This initiative employs the latest state-of-the-art image generation and processing technology to enhance user experience and elevate quality of generated videos.

Our paper covering our work can be found [here](Chivileva_Lynch_FinalYearProject.pdf).

## About the Project

This project can be broken down into four main parts, the first is the Text-to-Video (T2V) algorithms, these are used to generate new videos based on a text prompt. This is followed by a Video-to-Text (V2T) model that is used to generate captions for the aforementioned generated videos. A combination of tools are then used to analysis the video naturalism and the semantic matching. 

The code for the T2V models can be found in the directory [T2V Models](T2V_models), this includes instructuctions on how to setup the neccessary enviroment and how to run each of the models that we use for evaluation.  

The next part is the V2T model this is stored in [Video Caption Generator](video_caption_generator) directory, the model that we use is BLIP, we use it to generate a caption for each frame in a video for further analysis.
![Metric Workflow](assets/metric_ensemble2.png)
The metric workflow shown above, can be divided into two parts.
The first part involves data generation, depicted in blue and yellow boxes on the left-hand side of the figure. Starting with an initial text prompt, we generate a video using a T2V model. Then, we use the generated video to produce a list of captions using BLIP-2.
The second part involves the ensemble of three metrics, which starts with the Text Similarity Metric presented in [Text Similarity](Textual_Semantic_Similarity). This metric calculates the similarity score between the original text prompt and the BLIP-generated captions, ranging from 0 to 1. Next, we use the Naturalness Metric described in [Video Naturalness](Video_Naturalness) section, a customised XGBoost classifier that takes the generated video as input and outputs a score ranging from 0 to 1. Lastly, Average Precision presented in [GDINO](GDINO) section is calculated using the objects in the video based on the original prompt and the BLIP-generated captions, with a possible range of 0 to 1.
In order to aggregate the evaluation metrics, a weighted average based on a linear regression(LR) model presneted in [LinearRegression notebook](LinearRegression.ipynb) that was trained using manually rated videos is employed. This approach enables us to incorporate variations in each metric, as well as any potential biases or inconsistencies that could emerge from using a single metric. It is important to note that this technique has limitations, particularly in terms of the dataset size used to train the LR model and our own human biases. To address these limitations in future work, we aim to broaden the number of videos used for evaluation and to include outside human evaluation.

---
## Project Objectives
- [x]    Generate 35 unique videos for each T2V model based on the prompts in the file "/generated_videos/prompts.txt"
- [x] Use BLIP to caption the frames of the generated videos in the "generated_videos/" directory
- [x] Find a way to accurately interpret non-natural videos that are not easily comprehensible by humans
    - [x] Assess existing methods for measuring image naturalness
    - [x] Investigate the statistical properties of images that can impact their naturalness and aid in distinguishing non-interpretable images
    - [x] Modify and implement inception score to evaluate the quality of generated images
    - [x] Produce numeric data that represents videos and integrates all methodologies
    - [x] Create a custom-build image naturalness classifier
- [x] Caculate the IoU (Intersect over Union) and AP(Average Precision) of the frames of a video based on the original captions used to generate the videos and the BLIP generated captions using Grouning DINO (GDINO) for zero-shot object detction.
- [X] Combine the naturalism and semantic matching metrics to create our novel metric.
- [ ] Evaluate the accuracy of the model aganist human evaluation.
- [ ] Increase the number of prompts used to 100.
- [ ] Create a tool to explain areas where the generated video fails to capture the original caption. 
---
## Project Highlights
The study highlights the constraints of the current evaluation metrics used in the literature, identifying the need for a new evaluation metric that addresses these limitations. Our main contribution is a proposed evaluation metric which involves addressing two critical challenges: image naturalness and semantic matching. Our research demonstrates that the novel metric outperforms a commonly used metric, indicating that it is a reliable and valid evaluation tool.

### Results

The comprehensive outcomes of the experimentation conducted on 35 distinct text formats and 5 distinct T2V models are presented in the [Results](Results.xlsx). [Research Paper](Chivileva_Lynch_FinalYearProject.pdf) provides a detailed explanation of the project is available for review.


---
### Recommended Steps:

1. SSH into a GPU (Needed to run T2V models)
2. Create a Docker Image using the "./Dockerfile"
    1. Configure {PORT1} for the notebook
3. Run image by running the script "./run_image.sh"
    1. Configure {PORT1}, {PORT2} to port forward the notebook and {UID} to create multiple containers from one image.
3. When the Docker Container is running:
    1. Run the file [setup.sh](setup.sh).
4. To create new videos using the T2V models:
    1. Go to the "./T2V_models/$MODEL_TYPE" directory
    2. Then run a model of your choice using the instructions inside the README.md file.
5. Generate captions for the videos with the "./video_caption_generator" directory.
    1. This model uses BLIP to generate captions for each frame in a video.
    2. The sets to use the model can be found inside the README.md file.
6. Generate Text Similarity Score by running [text_similarity.py](Textual_Semantic_Similarity/text_similarity.py)
7. Generate Image Naturalness score by following instructions from [Video_Naturalness](Video_Naturalness)
8. Generate AP IoU score by following instructions from [GDINO](GDINO)

---
## Project Directory Structure
A majority of the directories include a more detailed README.md that explains how to run the code and what it is used for.
They also included a ./setup.sh script if they are needed to run the code.
```
./
├── .git
├── .gitignore
├── GDINO
|    ├── Model.py (Script to run Grounding DINO on the generated videos)
|    ├── GroundingDINO/ (Directory containing all the code need to run the model)
|    └── AveragePrecision.py (Script used to calculate the Average Precision of a video)
├── T2V_models
|    └── $MODEL_TYPE (A directory for each T2V model used)
|        └── Model script (The Python code used to run the model)
├── Textual_Semantic_Similarity
|    └── text_similarity.py (Python script to calculate a textual semantic similarity for a generated video)
├── Video_Naturalness
|    └── Classifier (A folder that containes experiments and trained classifiers for image naturalness detection)
|    └── Inception (**Philip**)
|    └── image_statistics.py (Python script with functions to extract statistical properties of an image)
|    └── video_processing.py (Python script to generate naturalness scores for videos)
├── etc
|    └── Conda shell script used to manage python virtual enviroments
├── generated_videos
|    ├── $MODEL_TYPE (The output of the T2V models)
|    └── prompts.txt (The prompts used to generate the videos)
├── video_caption_generator
|    ├── video_caption_generator.py (Generates captions for the frames of videos)
|    └── video_captions.json (All the videos captions are stored here)
├── Dockerfile (Used to create the Docker Image used to run the code)
├── run_image.sh (Used to create the Docker Container)
└── setup.sh (used to create the inital enviroment needed to run all the code)


```
 
