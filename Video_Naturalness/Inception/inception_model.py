# OS and Sys
import sys
import os

# Data manipulation
import numpy as np
import pandas as pd

# Inception Model and Image Processing
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# Import functions from other files
from .utils import create_image_list, recalculate_class_prob

class Inception_Model():

    """
    This predicts the class of the frames of a video using the inception model and saves
    the average probability for the most probable class.
    """

    def __init__(self, input_filepath:str, output_filepath:str, use_modified:str) -> None:
        self.input_filepath = input_filepath # File(s) for video(s)
        self.output_filepath = output_filepath # CSV file to output results to
        self.results = {"video_name":[], "inception":[]}

        self.modified_inception_model = True if use_modified == "True" else False

        self.model = InceptionV3() # Model train on ImageNet

        if self.input_filepath.endswith(".mp4") or self.input_filepath.endswith(".gif"): # Single video evaluation
            self.process_video(self.input_filepath)
        elif os.path.isdir(self.input_filepath): # Multiple videos evaluation
            for video_file in os.listdir(path=self.input_filepath):
                if video_file.endswith(".mp4") or video_file.endswith(".gif"):
                    print(video_file)
                    self.process_video(f"{self.input_filepath}{video_file}")
        else:
            print(f"{self.input_filepath} isn't the file format mp4 or gif or a directory")
            exit()

        pd.DataFrame(self.results).to_csv(self.output_filepath, index=False)

    def process_video(self, video_path:str) -> None:
        # Extract each frame, process them then save to a list
        frames = create_image_list(video_path)
    
        # Calculate the inception score for the current video
        inception = self.calculate_inception_probability(frames)

        self.results["video_name"].append(video_path.split("/")[-1].split(".")[0])
        self.results["inception"].append(inception)
    
    # assumes images have any shape and pixels in [0,255]
    def calculate_inception_probability(self, images: np.ndarray) -> float: 
        # pre-process images, scale to [-1,1]
        images = preprocess_input(images)
        # predict p(y|x)
        p_yx = self.model.predict(images)
        # return the mean of each class throughout the frames of the video
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        if self.modified_inception_model:
            ids = np.argsort(p_y[0])[::-1]
            p_yx = p_yx[:,ids]
            class_names_sim_path = "imagenet_classes/imagenet_class_simscore.csv"
            p_yx = recalculate_class_prob(ids, class_names_sim_path, p_yx)
            p_y = np.expand_dims(p_yx.mean(axis=0), 0)
        
        # calculate the max mean probability
        max_class_probability = np.max(p_y.flatten())
        return max_class_probability

if __name__ == "__main__":
    len_argv = len(sys.argv)
    """
    The three flags:
    - fp: Stands for Use filepath, this is a string of the path of the video or directory of videos
    - o: The output file that the video(s) inception score is saved to.
    - m: 
    """
    flags = {"-fp":"videos/", "-o":"results/inception_model.csv", "-m":"False"}

    if len_argv % 2 == 1:
        for i in range(1,len_argv,2):
            if sys.argv[i] in flags.keys():
                flags[sys.argv[i]] = sys.argv[i+1]
            else:
                print(f"Found {sys.argv[i]}, file is looking for -fp for filepath or -o for output file.")
                exit()
    else:
        print("Missing Argument")
        print("\n".join(sys.argv[1:]))
        exit()

    Inception_Model(input_filepath=flags["-fp"], 
                    output_filepath=flags["-o"],
                    use_modified = flags["-m"], 
                    )
    
