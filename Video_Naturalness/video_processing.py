from image_statistics import *
from Inception.inception_score import Inception_Score
from Inception.inception_model import Inception_Model
import skvideo.io
import skvideo.utils
import os
import numpy as np
import matplotlib.pyplot as plt
import pyiqa
import torch
from skimage.feature import ORB
from skimage.color import rgba2rgb, rgb2gray
from skimage.io import imread
import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageChops
niqe_metric = pyiqa.create_metric('niqe').cuda()
brisque_metric = pyiqa.create_metric('brisque').cuda()
from skimage import feature, measure, color
import math
from sklearn.cluster import KMeans
import random
from skimage import io as skio
from skimage import color
import pandas as pd
import torch
import pickle
import xgboost
import warnings 
warnings.filterwarnings("ignore", message="some warning message") # remove warnings

def process_videos(video_path:str) -> pd.DataFrame:
  '''
  Extracts features from a single video or all videos in a directory.
  Args:
     video_path: str
       The path to the video file or directory containing videos to process.
  Returns:
     pd.DataFrame
       A dataframe with the extracted features for each video.
  '''    
  # check if it as a single video or a directory with videos
  if os.path.isfile(video_path):
    _, file_ext = os.path.splitext(video_path)
    if file_ext.lower() not in ['.mp4', '.gif']:
      print("Unsupported file format. Please provide a .mp4 or .gif file.")
      return None
    file_list = [video_path]   
  elif os.path.isdir(video_path): # multiple videos
    file_list = [os.path.join(video_path, file) for file in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, file))]
  # check that all videos are in .gif or .mp4 format
    for file in file_list:
      _, file_ext = os.path.splitext(file)
      if file_ext.lower() not in ['.mp4', '.gif']:
        print(f"Unsupported file format: {file}. Please provide only .mp4 or .gif files.")
        return None
  else: # file does not exist
    print(f'{video_path} is not a valid file or directory')
    pass
  # to store mean values for each video
  niqe_vid, niqe_v_vid, niqe_y_vid, niqe_u_vid, brisque_vid = [], [], [], [], []
  sharpness_vid, entropy_vid, spectral_vid, kp_std_vid, contrast_vid = [], [], [], [], []
  num_blobs_vid, avg_size_blobs_vid, texture_mean_vid, texture_std_vid, color_dist_vid = [], [], [], [], []
  kp_dist_mean_vid, desc_len_mean_vid, desc_len_std_vid = [], [], []
  incept_score_0, incept_score_1, incept_model, vid_names = [], [], [], []
  current_dir = os.getcwd()
  filename_yuv = os.path.join(current_dir, "misc/test.yuv")
  for video_file in file_list:
    niqe_vals, niqe_v_vals, niqe_y_vals, niqe_u_vals, brisque_vals = [], [], [], [], []
    sharpness_vals, entropy_vals, spectral_vals = [], [], []
    kp_std_vals, contrast_vals, kp_dist_mean_vals = [], [], []
    num_blobs_vals, avg_size_blobs_vals, desc_len_mean_vals, desc_len_std_vals = [], [], [], []
    texture_mean_vals, texture_std_vals, color_dist_vals= [], [], []
    try:
      i_s_0 = Inception_Score(input_filepath=video_file, output_filepath=os.path.join(current_dir, "misc/res.csv") , model_type=0)
      i_s_1 = Inception_Score(input_filepath=video_file, output_filepath=os.path.join(current_dir, "misc/res.csv"), model_type=1)
      i_m = Inception_Model(input_filepath=video_file, output_filepath=os.path.join(current_dir, "misc/res.csv"), use_modified=True)
      incept_score_0.append(i_s_0.results['inception_score'][0])
      incept_score_1.append(i_s_1.results['inception_score'][0])
      incept_model.append(i_m.results['inception'][0])
      vid_names.append(i_s_0.results['video_name'][0])
    except:
      incept_score_0.append(np.nan)
      incept_score_1.append(np.nan)
      incept_model.append(np.nan)
      vid_names.append(video_file)
      pass
    # statistical proprties
    vid = skvideo.io.vread(video_file)
    T, M, N, C = vid.shape
    print('The number of frames is: ', T)
    # produces a yuv file using -pix_fmt=yuvj444p
    skvideo.io.vwrite(filename_yuv, vid)
    # now to demonstrate YUV loading
    vid_luma = skvideo.io.vread(filename_yuv, height=M, width=N, outputdict={"-pix_fmt": "gray"})[:, :, :, 0]
    vid_luma = skvideo.utils.vshape(vid_luma)
    vid_rgb = skvideo.io.vread(filename_yuv, height=M, width=N)
    # now load the YUV "as is" with no conversion
    vid_yuv444 = skvideo.io.vread(filename_yuv, height=M, width=N, outputdict={"-pix_fmt": "yuvj444p"})
    # re-organize bytes, since FFmpeg outputs in planar mode
    vid_yuv444 = vid_yuv444.reshape((M * N * T * 3))
    vid_yuv444 = vid_yuv444.reshape((T, 3, M, N))
    vid_yuv444 = np.transpose(vid_yuv444, (0, 2, 3, 1))
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_file)
    keep_frames = T # can be adjusted if the execution time is too long
    frame_numbers = random.sample(range(T), keep_frames) # the number of random frames to extract
    # Extract the frames and save them as images
    for i, frame_number in enumerate(frame_numbers):
        # Set the frame number to extract
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        # Read a frame from the video
        ret, colour_frame = cap.read()
        if ret:
          colour_image_file = os.path.join(current_dir, "misc/colour_frame.jpg") # colour image
          # keep this as well to avoid mistakes
          cv2.imwrite(colour_image_file, colour_frame)
            # if the video is make-a-video or imagen removing water marks
            # 25 for make-a-video and 45 for imagen # if no just keep at 0
          model_name = video_file.split('/')[-2]
          crop_val = 0
          if model_name == 'Original_Videos_Meta_Google':
            extension = video_file[-3:]
            if extension == 'mp4': # imagen
                  crop_val = 45
            elif extension == 'gif': # make-a-video
                  crop_val = 25
          crop_image(colour_image_file, crop_val)
          colour_image = Image.open(colour_image_file)
          # Read a frame from the video
          frame = vid_luma[i]
          y_channel_frame =  vid_yuv444[i, :, :, 0]
          u_channel_frame =  vid_yuv444[i, :, :, 1]
          v_channel_frame =  vid_yuv444[i, :, :, 2]
          y_frame_image = os.path.join(current_dir, "misc/y_frame.jpg")
          u_frame_image = os.path.join(current_dir, "misc/u_frame.jpg")
          v_frame_image = os.path.join(current_dir, "misc/v_frame.jpg")
          image_file = os.path.join(current_dir, "misc/frame.jpg") # gray image
          cv2.imwrite(image_file, frame)
          cv2.imwrite(y_frame_image , y_channel_frame)
          cv2.imwrite(u_frame_image , u_channel_frame)
          cv2.imwrite(v_frame_image , v_channel_frame)
          # calculating all statistics of a frame
          texture_mean, texture_std = texture_score(colour_image_file)
          num_blobs, avg_size, blobs_score = detect_objects_blobs(colour_image_file)
          # storing in a list      
          try:
            niqe_vals.append(niqe_metric(image_file))
            niqe_v_vals.append(niqe_metric(v_frame_image))
            niqe_u_vals.append(niqe_metric(u_frame_image))
            niqe_y_vals.append(niqe_metric(y_frame_image))
            brisque_vals.append(brisque_metric(image_file))
          except:
              pass
          sharpness_vals.append(sharpness_score(colour_image_file))
          spectral_vals.append(spectral_score(colour_image_file))
          entropy_vals.append(entropy_score(image_file))
          contrast_vals.append(contrast_score(image_file))
          try:
            kp_dist_mean, kp_dist_std, desc_len_mean, desc_len_std =  compute_keypoint_stats(image_file)
            kp_dist_mean_vals.append(kp_dist_mean)
            kp_std_vals.append(kp_dist_std)
            desc_len_mean_vals.append(desc_len_mean)
            desc_len_std_vals.append(desc_len_std)
          except RuntimeError: # someties, when a video like a noise can not detect any key points 
            kp_std_vals.append(0)
            kp_dist_mean_vals.append(0)
            desc_len_mean_vals.append(0)
            desc_len_std_vals.append(0)
            pass
          num_blobs_vals.append(num_blobs)
          avg_size_blobs_vals.append(avg_size)
          texture_mean_vals.append(texture_mean)
          texture_std_vals.append(texture_std)
          color_dist_vals.append(colour_distribution_score(colour_image_file))
          # Display the Pillow Image object
        else:
          print("Error: Failed to capture a frame from the video source")
      # recording mean score for each video
    niqe_vals_float = [tensor.item() for tensor in niqe_vals]
    niqe_u_vals_float = [tensor.item() for tensor in niqe_u_vals]
    niqe_v_vals_float = [tensor.item() for tensor in niqe_v_vals]
    niqe_y_vals_float = [tensor.item() for tensor in niqe_y_vals]
    brisque_vals_float = [tensor.item() for tensor in brisque_vals]
    # adding mean values (average of all frames)    
    niqe_vid.append(sum(niqe_vals_float) / len(niqe_vals_float))
    niqe_v_vid.append(sum(niqe_v_vals_float) / len(niqe_v_vals_float))
    niqe_y_vid.append(sum(niqe_y_vals_float) / len(niqe_y_vals_float))
    niqe_u_vid.append(sum(niqe_u_vals_float) / len(niqe_u_vals_float))
    brisque_vid.append(sum(brisque_vals_float) / len(brisque_vals_float))
    sharpness_vid.append(sum(sharpness_vals) / len(sharpness_vals))
    entropy_vid.append(sum(entropy_vals) / len(entropy_vals))
    spectral_vid.append(sum(spectral_vals) / len(spectral_vals))
    kp_std_vid.append(sum(kp_std_vals) / len(kp_std_vals))
    contrast_vid.append(sum(contrast_vals) / len(contrast_vals))
    num_blobs_vid.append(sum(num_blobs_vals) / len(num_blobs_vals))
    avg_size_blobs_vid.append(sum(avg_size_blobs_vals) / len(avg_size_blobs_vals))
    texture_mean_vid.append(sum(texture_mean_vals) / len(texture_mean_vals))
    texture_std_vid.append(sum(texture_std_vals) / len(texture_std_vals))
    color_dist_vid.append(sum(color_dist_vals) / len(color_dist_vals))
    kp_dist_mean_vid.append(sum(kp_dist_mean_vals) / len(kp_dist_mean_vals))
    desc_len_mean_vid.append(sum(desc_len_mean_vals) / len(desc_len_mean_vals))
    desc_len_std_vid.append(sum(desc_len_std_vals) / len(desc_len_std_vals))
  # create a dictionary that maps column names to lists
  data = {'niqe': niqe_vid, 'niqe_v': niqe_v_vid, 'niqe_y': niqe_y_vid, 'niqe_u' : niqe_u_vid,
          'brisque' : brisque_vid, 'sharpness' :sharpness_vid, 'entropy' : entropy_vid, 'spectral' :spectral_vid,
          'kp_std' : kp_std_vid, 'contrast_vals' : contrast_vid, 'num_blobs' : num_blobs_vid,
          'avg_size_blobs' : avg_size_blobs_vid, 'texture_mean' : texture_mean_vid,
          'texture_std' : texture_std_vid, 'color_dist' : color_dist_vid, 'kp_dist_mean' : kp_dist_mean_vid, 
          'desc_len_mean' : desc_len_mean_vid, 'desc_len_std' : desc_len_std_vid,
          'inception_score_0': incept_score_0, 'inception_score_1': incept_score_1, 'inception_model' : incept_model, 'video_name' : vid_names
          }
  # create a DataFrame to store all extracted information for each video
  df = pd.DataFrame(data)
  # save the dataframe to a CSV
  df.to_csv(os.path.join(current_dir, "misc/video_statistics.csv"), index=False)
  return df

def main():
    '''
    To run the script from terminal. Naturalness score for video(s).
    Results will be stored in naturalness_results.csv
    '''
    # Get path to video file from user input
    file_path = input("Please enter the path to the video file: ")

    # Process the video file to extract features
    data = process_videos(file_path)

    # Load the trained XGBoost classifier
    current_dir = os.getcwd()
    model_path = os.path.join(current_dir, "Classifier/xgb_model_last.pkl")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

    # Make predictions on the video features using the loaded model
    new_df = data.iloc[:, :-1]
    confidence_scores = loaded_model.predict_proba(new_df)[:, 0]

    # Create a dataframe with the video names and confidence scores
    video_names = data['video_name']
    results_df = pd.DataFrame({'video_name': video_names, 'confidence_score': confidence_scores})

    # Save the results to a CSV file
    results_path = os.path.join(current_dir, "naturalness_results.csv")
    results_df.to_csv(results_path, index=False)

    # Print the confidence scores to the console
    print(f"The confidence scores for the videos in {file_path} are:\n{confidence_scores}")


if __name__ == "__main__":
    main()
       