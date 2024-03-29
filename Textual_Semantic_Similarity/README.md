### Textual Semantic Similarity
The purpose of this metric is to evaluate the textual semantic similarity between the generated video caption and the original caption. In our approach, we have decided to combine BERT and Cosine similarity. Combined similarity score reflects both the surface-level and deeper semantic similarities between the two sentences, thus providing a more accurate representation of their overall similarity. We opted to calculate the weighted textual similarity for generated video. The weights are assigned based on the frequency of each caption in the overall list of generated captions. 

### Enviroment Setup

You can run the [setup.sh](setup.sh) script but we suggest to create a Python virtual environment to run the code.
We also suggestion using conda to mange the enviroment, using the following commands.
```
conda init bash
conda create -n Text_Similarity anaconda python=3.9
conda activate Text_Similarity
ipython kernel install --user --name=Text_Similarity

apt-get update
apt-get install -y libgl1-mesa-glx
apt-get install libglib2.0-0

#installing all necessary packages
pip3 install transformers
pip install spacy
python3 -m spacy download en_core_web_sm
pip install nltk
```
### Running the Script

After you have properly set up the enviroment you can run the Python script [text_similarity.py](text_similarity.py)
The user will be prompted to provide a path to the json file that contains captions generated for each frame. To generate captions for a video follow instructions from (video_caption_generator)[video_caption_generator]. With this script you can generate a textual similarity score for one video only.

---
### Output
The output of the script video_processing.py is saved to the CSV file **video_text_similarity.csv**
Contained within this CSV file are the original caption of the video, the generated by BLIP caption with the highest similarity score and the weighted similarity score for the video.