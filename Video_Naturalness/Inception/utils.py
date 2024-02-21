import numpy as np
import pandas as pd

import cv2
from skimage.transform import resize

# scale an array of images to a new size
def scale_images(images, new_shape):
    images_list = list()
    for image in images:
        # resize with nearest neighbor interpolation
        new_image = resize(image, new_shape, 0)
        # store
        images_list.append(new_image)
    return np.asarray(images_list)

def create_image_list(filepath, frame_cap=50):
    vidcap = cv2.VideoCapture(filepath)
    success,raw_image = vidcap.read()
    images = []
    count = 1
    while success and count <= frame_cap:
        count +=1
        images.append(raw_image)
        success,raw_image = vidcap.read()
    print(f"The video had {count} frames")

    images = np.asarray(images)
    # convert from uint8 to float32
    images = images.astype('float32')
    # scale images to the required size
    images = scale_images(images, (299,299,3))
    return images

def reduce_classes(ids, class_names_sim):
    new_classes, used_values = {}, {}
    used_classes = []
    for i in range(len(ids)):
        if i not in used_classes:
            used_classes.append(i)
            new_classes[i] = []
        for j in range(i+1,len(ids)-1):
            if class_names_sim[ids[i],ids[j]] >= 0.75 and (j not in used_classes):
                if i in used_values.keys():
                    try:
                        new_classes[used_values[i]].append(j)
                        used_values[j] = used_values[i]
                    except:
                        if j not in used_classes:
                            print(i, used_values[i], new_classes)
                else:
                    new_classes[i].append(j)
                    used_values[j] = i

                used_classes.append(j)
    
    return new_classes

def recalculate_class_prob(ids, class_names_sim_path, p_yx):
    class_names_sim = pd.read_csv(class_names_sim_path).to_numpy().astype("float32")
    new_classes = reduce_classes(ids, class_names_sim)
    for keep_index in new_classes:
        for remove_index in new_classes[keep_index]:
            p_yx[:,keep_index] += p_yx[:,remove_index]
            p_yx[:,remove_index] = 0
    return p_yx

def smooth_frame_probability(p_yx, marginal_dist):
    for ith_frame, frame in enumerate(p_yx):
        if np.max(frame) > 0.75:
            max_index = np.argmax(frame)
            p_yx[ith_frame] = [1E-16]*len(marginal_dist)
            p_yx[ith_frame][max_index] = 1.0 - (1E-16*(len(marginal_dist)-1))
        elif np.max(frame) < 0.10:
            p_yx[ith_frame] = marginal_dist
    
    return p_yx