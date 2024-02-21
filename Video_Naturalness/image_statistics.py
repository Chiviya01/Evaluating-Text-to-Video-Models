import os
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
    
def crop_image(image_path:str, crop_height:int):
    '''
    This function is used to crop the bottom of the image, to remove water marks
    '''
    # Load the image
    image = Image.open(image_path)
    # Get the width and height of the image
    width, height = image.size
    # Define the region to crop (bottom portion)
    left = 0
    upper = 0
    right = width
    lower = height - crop_height
    # Crop the image
    cropped_image = image.crop((left, upper, right, lower))
    # Save the cropped image
    cropped_image = cropped_image.convert('RGB')
    cropped_image.save("colour_frame.jpg")
    # Display the cropped image
    #cropped_image.show()

# analysing the texture of an image 
def texture_score(image_path:str):
  '''
  This function measures the degree of uniformity in the texture of an image, 
  by calculating the variance of the image's gradient magnitude.
  '''
  # Load the image
  img = cv2.imread(image_path)
  # Convert the image to grayscale
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Apply a Gaussian blur to reduce noise
  blur = cv2.GaussianBlur(gray, (5, 5), 0)
  # Apply Sobel edge detection in x and y direction
  sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
  sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
  # Compute the magnitude of the gradient
  mag = np.sqrt(np.square(sobelx) + np.square(sobely))
  # Compute the mean and standard deviation of the gradient magnitude
  mean = np.mean(mag)
  std = np.std(mag)
  # Print the results
  # print('Mean:', mean)
  # print('Standard Deviation:', std)
  return mean, std

def detect_objects_blobs(image_path:str):
  '''
  This funtion detects blobs or circular objects in an image using the Laplacian of Gaussian (LoG) method.
  '''
  # Load the image
  image = Image.open(image_path)
  # Convert the image to grayscale
  gray_image = np.array(image.convert('L'))
  # Detect blobs in the image
  blobs = feature.blob_dog(gray_image, min_sigma=1, max_sigma=50, threshold=0.1)
  # Get the number of blobs and average size
  num_blobs = len(blobs)
  if num_blobs > 0:
      avg_size = sum([blob[2] for blob in blobs]) / num_blobs
  else:
      avg_size = 0
  # Calculate the score based on the number and size of blobs
  if num_blobs == 0:
      score = 0
  elif num_blobs == 1 and avg_size > 300:
      score = 10
  elif num_blobs == 1 and avg_size <= 300:
      score = 5
  elif num_blobs > 1 and avg_size > 300:
      score = 8
  else:
      score = 3
  # Print the results
  # print('Number of blobs:', num_blobs)
  # print('Average blob size:', avg_size)
  # print('Score:', score)
  return num_blobs, avg_size, score

def compute_keypoint_stats(image_path:str):
  '''
  This funtion computes statistics about keypoints in an image, including the mean, standard deviation, and density of keypoints
  '''
  # need to pass a 2-d image 
  image = imread(image_path)
  # Detect ORB features
  descriptor_extractor = ORB(n_keypoints=200)
  descriptor_extractor.detect_and_extract(image)
  keypoints = descriptor_extractor.keypoints
  # Plot the keypoints on the original image
  # fig, ax = plt.subplots()
  # ax.imshow(image)
  # ax.scatter(keypoints[:, 1], keypoints[:, 0], c='r', marker='x')
  # plt.show()
  orb = ORB(n_keypoints=2000)
  orb.detect_and_extract(image)
  keypoints = orb.keypoints
  descriptors = orb.descriptors
  # Compute some statistics on the keypoints
  kp_dist = np.sqrt(np.sum(np.square(np.diff(keypoints, axis=0)), axis=1))
  kp_dist_mean = np.mean(kp_dist)
  kp_dist_std = np.std(kp_dist)
  # Compute some statistics on the descriptors
  desc_len = np.sum(descriptors, axis=1)
  desc_len_mean = np.mean(desc_len)
  desc_len_std = np.std(desc_len)
  return kp_dist_mean, kp_dist_std, desc_len_mean, desc_len_std
# kp_dist_mean: the mean distance between keypoints in the image.
# kp_dist_std: the standard deviation of the distances between keypoints in the image.
# desc_len_mean: the mean length of the descriptors associated with the keypoints in the image.
# desc_len_std: the standard deviation of the lengths of the descriptors associated with the keypoints in the image.

def contrast_score(image_path:str) -> float:
  '''
  This function measures the difference between the lightest and darkest parts of an image
  '''
  # calculating cotrast score of an image
  # need to pass gray image
  gray_image = imread(image_path)
  # Convert the image to a NumPy array
  gray_array = np.array(gray_image)
  # Calculate the standard deviation of the pixel intensities
  std_dev = np.std(gray_array)
  # Calculate the contrast score as the standard deviation divided by the mean intensity
  mean_intensity = np.mean(gray_array)
  contrast_score = std_dev / mean_intensity
  return contrast_score

def entropy_score(image_path:str) -> float:
  '''
  This function measures the randomness or disorder of pixel values in an image
  '''
  # need to pass gray image
  gray_image = imread(image_path)
  # Convert the image to a NumPy array
  gray_array = np.array(gray_image)
  # calculating image entropy
  # Calculate the image histogram
  histogram = np.histogram(gray_array, bins=256)[0]
  # Calculate the probability of each intensity level
  probabilities = histogram / float(np.sum(histogram))
  # Calculate the entropy of the histogram
  entropy = -np.sum([p * math.log(p, 2) for p in probabilities if p > 0])
  return entropy

def spectral_score(image_path:str) -> float:
  '''
  This function measures the degree to which an image deviates from natural image statistics in the Fourier domain.
  '''
  # passing colour image
  colour_image = imread(image_path)
  # getting a single value score for spectral analysis
  # Convert the image to a numpy array
  image_array = np.array(colour_image)
  # Calculate the mean and standard deviation of each color channel
  r_mean = np.mean(image_array[:,:,0])
  g_mean = np.mean(image_array[:,:,1])
  b_mean = np.mean(image_array[:,:,2])
  r_std = np.std(image_array[:,:,0])
  g_std = np.std(image_array[:,:,1])
  b_std = np.std(image_array[:,:,2])
  # Calculate a spectral score as the sum of standard deviations divided by the sum of means
  spectral_score = (r_std + g_std + b_std) / (r_mean + g_mean + b_mean)
  return spectral_score

def colour_distribution_score(image_path:str) -> float:
  '''
  This function measures the uniformity of color in an image
  '''
  colour_image = Image.open(image_path)
  image_lab = color.rgb2lab(np.array(colour_image))
  # Extract the A and B channels from the LAB image
  a_channel, b_channel = image_lab[:,:,1], image_lab[:,:,2]
  # Flatten the A and B channels into a 2D array
  a_flat = a_channel.flatten().astype(np.float32)
  b_flat = b_channel.flatten().astype(np.float32)
  ab_flat = np.column_stack((a_flat, b_flat))
  # Apply K-means clustering with 2 clusters
  kmeans = KMeans(n_clusters=2, random_state=0).fit(ab_flat)
  # Find the cluster center with the lowest A channel value
  cluster_centers = kmeans.cluster_centers_
  cloud_cluster = np.argmin(cluster_centers[:,0])
  # Calculate the cloud distribution score as the proportion of pixels in the cloud cluster
  labels = kmeans.labels_.reshape(a_channel.size)
  cloud_pixels = np.sum(labels == cloud_cluster)
  total_pixels = labels.size
  cloud_score = cloud_pixels / total_pixels
  return cloud_score

def sharpness_score(image_path:str) -> float:
  '''
  This function measures the amount of high-frequency content in an image, which is indicative of the image's level of detail.
  '''
  colour_image = Image.open(image_path)
  # calculate a single score for an image sharpness
  # Apply a sharpening filter to the image
  filtered_image = colour_image.filter(ImageFilter.SHARPEN)
  # Calculate the difference between the original image and the filtered image
  diff = ImageChops.difference(colour_image, filtered_image)
  # Calculate the sharpness score as the root mean square (RMS) of the pixel values in the difference image
  sharpness = (np.array(diff).mean() ** 2) ** 0.5
  return sharpness