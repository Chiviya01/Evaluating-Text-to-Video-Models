# OS and Sys
import sys
import os

# Data manipulation
import numpy as np
import pandas as pd

# Word embedding and similarity score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Inception Model and Image Processing
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# Import functions from other files
from .utils import create_image_list, recalculate_class_prob, smooth_frame_probability

class Inception_Score():
    def __init__(self, input_filepath:str, output_filepath:str, model_type:int) -> None:
        self.input_filepath = input_filepath # File(s) for video(s)
        self.output_filepath = output_filepath # CSV file to output results to
        self.model_type = model_type # Indicate which model to use (IS or MIS1 or MIS2)
        self.results = {"video_name":[], "inception_score":[]}
        
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
        inception_score = self.calculate_inception_score(frames)

        self.results["video_name"].append(video_path.split("/")[-1].split(".")[0])
        self.results["inception_score"].append(inception_score)
    
    # assumes images have any shape and pixels in [0,255]
    def calculate_inception_score(self, images: np.ndarray, eps=1E-16) -> float: 
        # pre-process images, scale to [-1,1]
        images = preprocess_input(images)
        # predict p(y|x)
        p_yx = self.model.predict(images)
        # return the mean of each class throughout the frames of the video
        p_y = np.expand_dims(p_yx.mean(axis=0), 0)

        # Model type uses the unmodified inception score(not intended for videos)
        if self.model_type == 0:
            marginal_dist = p_y + eps
        # The MIS uses the ideal marginal distribution to achieve a larger score for the ideal frame distribution
        elif self.model_type > 0:
            marginal_dist = [1/len(p_y[0])]*len(p_y[0])
        # Combines the probability of similar classes into one class
        if self.model_type > 1:
            ids = np.argsort(p_y[0])[::-1]
            p_yx = p_yx[:,ids]
            class_names_sim_path = "imagenet_classes/imagenet_class_simscore.csv"
            p_yx = recalculate_class_prob(ids, class_names_sim_path, p_yx)
        # Rewards low entropy frames and penalizes high entropy frames
        if self.model_type == 3:
            p_yx = smooth_frame_probability(p_yx, marginal_dist)
        # calculate KL divergence using log probabilities
        kl_d = p_yx * (np.log(p_yx + eps) - np.log(marginal_dist))
        # sum over classes
        sum_kl_d = kl_d.sum(axis=1)
        # average over images
        avg_kl_d = np.mean(sum_kl_d)
        # undo the log
        is_score = np.exp(avg_kl_d)
        return is_score

if __name__ == "__main__":
    len_argv = len(sys.argv)
    """
    The three flags:
    - fp: Stands for Use filepath, this is a string of the path of the video or directory of videos
    - o: The output file that the video(s) inception score is saved to.
    """
    flags = {"-fp":"videos/", "-o":"results/inception_score.csv", "-mt":0}

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

    Inception_Score(input_filepath=flags["-fp"], 
                    output_filepath=flags["-o"],
                    model_type = int(flags["-mt"]),
                    )
    