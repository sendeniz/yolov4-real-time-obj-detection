import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import pandas as pd
from utils.augmentations import AugmentImage as apply
from utils.augmentations import AugmentBoundingBox as augmentbox
cv.setNumThreads(6)
      
def augment_data(image, bboxes,
                  image_size = 416,
                  p_scale = 1.0,  scale_factor = 1.09,
                  p_trans = 1.0, translate_factor = 0.09,
                  p_rot = 0.3, rotation_angle = 45.0,
                  p_shear = 0.3, shear_angle = 10.0,
                  p_hflip = 0.3, 
                  p_vflip = 0.0, 
                  p_mixup = 0.3, 
                  p_mosaic = 0.3,
                  p_hsv = 1.0, hgain = 0.1, sgain=0.9, vgain=0.9, 
                  p_grey = 0.1,
                  p_blur = 0.1, 
                  p_clahe = 0.1,  
                  p_cutout = 0.3, 
                  p_shuffle = 0.1,
                  p_post = 0.1, mode = 'test'): 
    """
    Apply various transformations to an input image.

    Args:
        image (str): Path to the input image.
        scale_translate_factor (float): Scaling and translation factor, default is 0.2 (20%).
        resize (int): Size to which the image should be resized into a squared image, default is 608.
        horizontal_flip_probability (float): Probability of horizontal flipping, default is 0.5.
        vertical_flip_probability (float): Probability of vertical flipping, default is 0.0.
        angle (int): Angle for rotation in degrees, default is 45 (45 degrees).
        rotation_probability (float): Probability of rotation, default is 0.25.
        shear_factor (int): Shear factor in degrees, default is 0.1 (10 degrees).
        shear_probability (float): Probability of shear transformation, default is 0.25.

    Returns:
        np.ndarray: Transformed image.
    """ 
    
    if mode == 'train':
        # convert rgb to bgr
        image = apply.rgb_to_bgr(image)
        # resize image
        image = apply.resize(image, image_size)
        
        height, width = image.shape[:2]
        # conver to opencv format for augmentations
        bboxes = augmentbox.yolo_to_opencv(bboxes, width, height)
        # scale
        image, bboxes = apply.scale(image, bboxes, scale_factor = scale_factor, p = p_scale)
        # translate
        image, bboxes = apply.translate(image, bboxes, translate_factor = translate_factor, p = p_trans)

        # horizontal flip image
        image, bboxes = apply.horizontal_flip(image, bboxes, p = p_hflip)

        # vertical flip image
        # not implemented
        # image = apply.vertical_flip(image, bboxes, p = p_vflip)

        image, bboxes = apply.rotate(image, bboxes, rotation_angle = rotation_angle, p = p_rot)
       
        image, bboxes = apply.shear(image, bboxes, shear_angle = shear_angle, p = p_shear)

        image = apply.posterize(image, p = p_post)

        image = apply.greyscale(image, p = p_grey)
        
        image = apply.blur(image, p = p_blur)
        
        image = apply.clahe(image, p = p_clahe)

        image = apply.cutout(image, p = p_cutout)

        image = apply.hsv_augment(image, hgain = hgain, sgain = sgain, vgain = vgain, p = p_hsv)

        image = apply.shuffle_channel(image, p = p_shuffle)

        # mosaic and mixup overwrites existing image and augmentations
        # above augments are applied in mosaic and mixup themselves
        # apply either mosaic or mixup
        mosaic_or_mixup = np.random.choice(["mosaic", "mixup"])
        if mosaic_or_mixup == "mosaic":
            image, bboxes = apply.mosaic(image, bboxes,
                                        p_scale, scale_factor,
                                        p_trans, translate_factor,
                                        p_rot, rotation_angle,
                                        p_shear, shear_angle,
                                        p_hflip,
                                        p_hsv, hgain, sgain, vgain,
                                        p_grey,
                                        p_blur,
                                        p_clahe,
                                        p_cutout,
                                        p_shuffle,
                                        p_post,
                                        annotations_csv = "data/coco/train_5Kexamples.csv", 
                                        img_dir = "data/coco/images/", 
                                        label_dir = "data/coco/labels/", p = p_mosaic)
        else:
            image, bboxes = apply.mixup(image, bboxes,
                                        p_scale, scale_factor,
                                        p_trans, translate_factor,
                                        p_rot, rotation_angle,
                                        p_shear, shear_angle,
                                        p_hflip,
                                        p_hsv, hgain, sgain, vgain,
                                        p_grey,
                                        p_blur,
                                        p_clahe,
                                        p_cutout,
                                        p_shuffle,
                                        p_post,
                                        annotations_csv = "data/coco/train_5Kexamples.csv", 
                                        img_dir = "data/coco/images/", 
                                        label_dir = "data/coco/labels/", p = p_mixup)

    elif mode == 'test':
        # convert rgb to bgr
        image = apply.rgb_to_bgr(image)
        # resize image
        image = apply.resize(image, image_size)
        
        height, width = image.shape[:2]
        # conver to opencv format for augmentations
        bboxes = augmentbox.yolo_to_opencv(bboxes, width, height)

    image = apply.resize(image, width)
    # convert brg to rgb
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # normalise # consistent with ImageNet backbone
    image = apply.normalize(image) #, mean = [0.485, 0.456, 0.406], std = [0.229, 0.244, 0.255])
    #image = Image.fromarray(image)

    # convert opencv pascalvoc format to yolo
    bboxes = augmentbox.opencv_to_yolo(bboxes, width, height)

    return image, bboxes
