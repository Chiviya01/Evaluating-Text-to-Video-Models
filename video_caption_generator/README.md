# Video Caption Generator Using BLIP

This directory is used to generator captions for each frame in generated video.
We provide the ./setup.sh file to setup the enviroment need to run the Python script video_caption_generator.py 

### Directory Structure
```
./
├── video_caption_generator.py (Python script to generate captions for video frames)
├── setup.sh (Optional setup to run Python script)
└── video_captions.json (Output of video_caption_generator.py)
```

---
## Enviroment Setup
You can skip the setup made through the ./setup.sh script but we suggest to create a Python virtual environment to run the code.
We also suggestion using conda to mange the enviroment, using the following commands.

```
$ conda create -n BLIPCaptions anaconda python=3.9
$ conda activate BLIPCaptions
$ ipython kernel install --user --name=BLIPCaptions

$ pip3 install salesforce-lavis
```

---
## Running the Script
After you have properly set up the enviroment you can run the Python script video_caption_generator.py
This takes the generated videos from the  "../generated_videos/$MODEL_TYPE", where $MODEL_TYPE is the open source T2V algorithm used to create the videos.
Each of the models use the same 35 captions to generate 35 unique videos.

---
## Output
The output of the script video_caption_generator.py is saved to the JSON file video_captions.json.
This file is structured in the following way:

```
./
├── MODEL_TYPE_{1}
|.             ├──VIDEO_NAME{1}
|.             |...            ├── CAPTION{1}
|...           └──VIDEO_NAME{M}|...
└── MODEL_TYPE_{N}             └── CAPTION{O}
```

Where each model has 35 videos each and each frame of the video has a caption generated using BLIP.
