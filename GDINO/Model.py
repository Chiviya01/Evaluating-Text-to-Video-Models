import os
import numpy as np
from typing import Tuple, List
from PIL import Image
import cv2
import supervision as sv
import json
from AveragePrecision import find_AP

def create_frame_list(filepath):
    """
    This function takes in a video filepath.
    Reads in the video frame by frame using cv2.
    Converts the frame into a GDINO suitable format.
    Returns a list of frames
    """
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    vidcap = cv2.VideoCapture(filepath)
    success,raw_image = vidcap.read()
    source_frames = []
    frames = []
    while success:
        frame = Image.fromarray(raw_image)
        frame_transformed, _ = transform(frame, None)
        frames.append(frame_transformed)
        source_frames.append(raw_image)
        success,raw_image = vidcap.read()
    return source_frames, frames

def generate_bounding_boxes(frames, captions, real_caption, BOX_TRESHOLD = 0.5, TEXT_TRESHOLD = 0.3):
    """
    This function takes in a list of frames in a BLIP suitable format.
    Generates a caption for each frame.
    Returns a dictionary with the key being the frame id and the value the caption for that frame.
    """
    DINO_outputs = {}
    ap_value = None
    for i, frame in enumerate(frames):
        DINO_outputs[i] = {}
        boxes, logits, phrases = predict(
        model=model, 
        image=frame, 
        caption=captions[str(i)][0], 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

        DINO_outputs[i]["BLIP_caption"] = {"boxes":boxes.tolist(), 
                                           "logits":logits.tolist(), 
                                           "phrases":phrases}
        
        boxes, logits, phrases = predict(
        model=model, 
        image=frame, 
        caption=real_caption, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )
        DINO_outputs[i]["real_caption"] = {"boxes":boxes.tolist(), 
                                           "logits":logits.tolist(), 
                                           "phrases":phrases}
        
        ap_value = find_AP(DINO_outputs[i]["real_caption"], DINO_outputs[i]["BLIP_caption"])
    return DINO_outputs, ap_value

def is_video(filename, filetypes=[".mp4", ".gif"]):
    """
    Checks if a file is a video based on the file extension.
    Returns True if the file extension matches the selected types else False.
    """
    for filetype in filetypes:
        if filename.endswith(filetype):
            return True, filename.split(filetype)[0]
    return False, None



def run_DINO_model(filepath, captions, real_caption):
    """
    This function is used to generate bounding boxes for the real and BLIP generated captions
    for each frame in a video.
    
    By loading in the videos as a GDINO suitable format with create_frame_list(filepath).
    Then finding the 
    """
    _, frames = create_frame_list(filepath)
    return generate_bounding_boxes(frames, captions, real_caption, BOX_TRESHOLD = 0.5, TEXT_TRESHOLD = 0.3)

def read_BLIP_captions(caption_path = "../../video_caption_generator/video_captions.json"):
    """
    This function is used to read in the BLIP generated captions for each video. 
    """
    
    # Opening JSON file
    f = open(caption_path)

    # returns JSON object as 
    # a dictionary
    captions = json.load(f)
    return captions

def save_json_file(path, data):
    """
    This function is used to save dictionaries to a json file.
    """
    
    output_file = open(file=path, mode="w", encoding="utf-8") 
    json.dump(data, output_file, indent=4)
    output_file.close()
    
if __name__ == "__main__":
    
    # Setting up model and enviroment
    HOME = os.getcwd()
    print(f"Current working directory {HOME}")
    CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    print(CONFIG_PATH, "; exist:", os.path.isfile(CONFIG_PATH))
    assert(os.path.isfile(CONFIG_PATH))
    WEIGHTS_NAME = "groundingdino_swint_ogc.pth"
    WEIGHTS_PATH = os.path.join(HOME, "weights", WEIGHTS_NAME)
    print(WEIGHTS_PATH, "; exist:", os.path.isfile(WEIGHTS_PATH))
    assert(os.path.isfile(WEIGHTS_PATH))
    
    os.chdir(f"{HOME}/GroundingDINO")
    
    # Loading in GDINO model
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import groundingdino.datasets.transforms as T
    model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    
    # Dictionary which are save to json at the end.
    complete_DINO_outputs = {}
    ap_values = {}
    
    # Location for all videos.
    video_directory = "../../generated_videos"
    captions = read_BLIP_captions()
    
    # Used for single videos.
    isvideo, real_caption = is_video(video_directory)
    
    if isvideo:
        complete_DINO_outputs[video_directory], ap_values[video_directory] = run_DINO_model(video_directory)
    else: # If it is a directory
        for model_type in os.listdir(video_directory):
            filepath = f"{video_directory}/{model_type}/"
            if model_type.find(".")==-1:
                complete_DINO_outputs[model_type] = {}
                ap_values[model_type] = {}
                print(model_type)
                for video_name in os.listdir(filepath):
                    isvideo, real_caption = is_video(video_name)
                    if isvideo and model_type in captions:
                        text_prompt = real_caption.replace("_", " ")
                        if text_prompt in captions[model_type]:
                            complete_DINO_outputs[model_type][text_prompt],  ap_values[model_type][text_prompt] = run_DINO_model(
                                filepath+video_name, 
                                captions[model_type][text_prompt], 
                                text_prompt)

    save_json_file("../output/bounding_boxes.json", complete_DINO_outputs)
    save_json_file("../output/ap_values.json", ap_values)    