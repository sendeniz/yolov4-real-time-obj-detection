import cv2 as cv
import numpy as np
import pandas as pd
import random
import os
from PIL import Image

cv.setNumThreads(6)
image_size = 608 # 416

class AugmentImage:
    @staticmethod
    def normalize(image, mean = [0, 0 , 0], std = [1, 1, 1], max_pixel_val = 255):
        image = image.astype(np.float32)
        image = (image - np.array(mean, dtype=np.float32) * max_pixel_val) / (np.array(std, dtype=np.float32) * max_pixel_val)
        return image

    def posterize(image,  num_bits = 4, p = 0.1):
        if np.random.rand() <= p:
            levels = 2 ** num_bits
            divisor = 256 // levels
            lut = np.arange(256, dtype=np.uint8) // divisor * divisor
            image = cv.LUT(image, lut)
        return image

    def greyscale(image, p = 0.1):
        if np.random.rand() <= p:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = cv.merge([image, image, image])
        return image

    def blur(image, p = 0.1):
        if np.random.rand() <= p:
            kernel_width = np.random.randint(3, 7)
            kernel_height = np.random.randint(3, 7)
            kernel_width = kernel_width + 1 if kernel_width % 2 == 0 else kernel_width
            kernel_height = kernel_height + 1 if kernel_height % 2 == 0 else kernel_height
            std_deviation = np.random.uniform(0.2, 1.0)
            image = cv.GaussianBlur(image, (kernel_width, kernel_height), std_deviation)
        return image

    def clahe(image, p = 0.1):
        if np.random.rand() <= p:
            b, g, r = cv.split(image)
            # apply clahe to each color channel independently
            clahe = cv.createCLAHE(clipLimit = 4.0, tileGridSize=(8, 8))
            b_clahe = clahe.apply(b)
            g_clahe = clahe.apply(g)
            r_clahe = clahe.apply(r)
            # merge the clahe-enhanced channels back together
            image = cv.merge([b_clahe, g_clahe, r_clahe])
        return image

    def cutout(image, p = 0.3):
        if np.random.rand() <= p:
            height, width = image.shape[:2]
            scales = [0.10] * 1 + [0.080] * 2 + [0.060] * 4 + [0.040] * 8 + [0.020] * 16  # image size fraction
            for s in scales:
                mask_h = np.random.randint(1, int(height * s))  # create random masks
                mask_w = np.random.randint(1, int(width * s))
                # box
                xmin = max(0, np.random.randint(0, width) - mask_w // 2)
                ymin = max(0, np.random.randint(0, height) - mask_h // 2)
                xmax = min(width, xmin + mask_w)
                ymax = min(height, ymin + mask_h)
                # apply random color mask
                image[ymin:ymax, xmin:xmax] = [np.random.randint(64, 191) for _ in range(3)]
        return image

    def hsv_augment(image, hgain = 0.1, sgain=0.9, vgain=0.9, p = 1.0):
        if np.random.rand() <= p:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV))
            dtype = image.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            image_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val))).astype(dtype)
            image = cv.cvtColor(image_hsv, cv.COLOR_HSV2BGR, dst=image)
        return image

    def scale(image, bboxes, scale_factor = 1.09, lower_bound = 1.0, p = 1.0, preserve_ratio = True):
        """
        Scales an image by a random factor bound below by the lower_bound. 

        Args:
        - image (numpy.ndarray):
            The input image to be scaled.
        - scale_factor (float):
            The maximum ratio to which the image is scaled. Default is 1.09.
        - lower_bound (float):
            The minimum ratio to which the image is scaled. Default is original image size: 1.00.

        Returns:
        - numpy.ndarray:
            The scaled image.
        """
        if np.random.rand() <= p:
            height, width =  image.shape[:2]
            if preserve_ratio == True:
                scale_x = np.random.uniform(lower_bound, scale_factor)
                scale_y = scale_x
            elif preserve_ratio == False:
                scale_x = np.random.uniform(lower_bound, scale_factor)
                scale_x = np.random.uniform(lower_bound, scale_factor)

            image = cv.resize(image, None, fx = scale_x, fy = scale_y)
            # compute scale augmented bounding boxes
            bboxes[:,:4] *= [scale_x, scale_y, scale_x, scale_y]
            # clip bounding boxes to be within image dimension
            # and drop bounding boxes based on min visbility of 0.4 compared to orginal
            bboxes = AugmentBoundingBox.clip(bboxes, img_size = height, min_visibility = 0.4)
            # create black image for padding if scaled below 1.0
            blankimg = np.zeros((height, width, 3), dtype=np.uint8) 
            y_lim = int(min(scale_y, 1) * height)
            x_lim = int(min(scale_x, 1) * width)
            # paste scale image on black background
            # if scaled down padded background will be visible
            blankimg[:y_lim,:x_lim,:] = image[:y_lim,:x_lim,:]
            image = blankimg
        return image, bboxes
    
    def translate(image, bboxes, translate_factor = 0.09, p = 1.0, preserve_ratio = False):
        """
        Translates an image by a random factor representing the percentage of the image dimensions.

        Args:
        - image (numpy.ndarray):
            The input image to be scaled.
        - scale_factor (float):
            The maximum percentage to which the image is translated. Default is 9%: 0.09.
            The minimum percentage to which the image is scaled is the negative scale_factor.
        Returns:
        - numpy.ndarray:
            The translated image.
        """
        if np.random.rand() <= p:
            lower_bound = - translate_factor
            height, width =  image.shape[:2]
            if preserve_ratio == False:
                translate_x_val = random.uniform(lower_bound, translate_factor)
                translate_y_val = random.uniform(lower_bound, translate_factor)
            elif preserve_ratio == True:
                translate_x_val = random.uniform(lower_bound, translate_factor)
                translate_y_val = translate_x_val
            # create blank image for padding
            blankimg = np.zeros((height, width, 3), dtype=np.uint8) 
            corner_x = int(translate_x_val * width)
            corner_y = int(translate_y_val * height)
            # change origin to the top-left corner of the translated box
            orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(height, corner_y + height), min(width, corner_x + width)]
            mask = image[max(-corner_y, 0):min(height, -corner_y + height), max(-corner_x, 0):min(width, -corner_x + width),:]
            blankimg[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
            # translate bounding boxes
            bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]
            # clip bounding boxes to be within image dimension
            # and drop bounding boxes based on min visbility of 0.4 compared to orginal 
            bboxes = AugmentBoundingBox.clip(bboxes, img_size = height, min_visibility = 0.4)
            image = blankimg
        return image, bboxes
    
    def horizontal_flip(image, bboxes, p = 0.3):
        if np.random.rand() <= p:
            image = cv.flip(image, 1)
            height, width = image.shape[:2]
            center_width = width / 2
            # since we flip horizontally only center width is needed
            bboxes[:,0] += 2 * (center_width - bboxes[:,0])
            bboxes[:,2] += 2 * (center_width - bboxes[:,2])
            bboxes_width = abs(bboxes[:,0] - bboxes[:,2])
            bboxes[:,0] -= bboxes_width
            bboxes[:,2] += bboxes_width
        return image, bboxes

    def vertical_flip(image, bboxes, p = 0.1):
        if np.random.rand() <= p:
            image = cv.flip(image, 0)
        return image, bboxes
    
    def shear(image, bboxes, shear_angle = 10.0, p = 0.3):
        """
        Sheares an image by a random factor represting the degrees.

        Args:
        - image (numpy.ndarray):
            The input image to be scaled.
        - shear_angle (float):
            The maximum degrees by which the image will be sheared. Default is 10 degrees.

        Returns:
        - numpy.ndarray:
            The scaled image.
        - sheared_width (float):
            The shared image width. 
        - shared_height (float):
             The shared image height. 
        """
        if np.random.rand() <= p:
            shear_angle = np.random.uniform(-shear_angle / 100, shear_angle / 100)
            height, width = image.shape[:2]
            if shear_angle < 0:
                image, bboxes = AugmentImage.horizontal_flip(image, bboxes)
            # shear matrix
            M = np.array([[1, abs(shear_angle), 0], [0, 1, 0]])
            new_width = width + abs(shear_angle * height)
            bboxes[:,0] += (bboxes[:,1] * abs(shear_angle)).astype(int) 
            bboxes[:,2] += (bboxes[:,3] * abs(shear_angle)).astype(int)
            
            image = cv.warpAffine(image, M, (int(new_width), width))
            if shear_angle < 0:
                image, bboxes = AugmentImage.horizontal_flip(image, bboxes)

            # rotation changes height width of image
            # adjust to original square size
            image = AugmentImage.resize(image, width)
            scale_x = new_width / width
            bboxes[:,:4] /=  [scale_x, 1, scale_x, 1]
        return image, bboxes

    def rotate(image, bboxes, rotation_angle = 45, p = 0.3):
        if np.random.rand() <= p:
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            rotation_angle = np.random.uniform(-rotation_angle, rotation_angle) 
            # define rotation matrix
            M = cv.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            # new width and height of rotated image
            rotated_width = int((height * sin) + (width * cos))
            rotated_height = int((height * cos) + (width * sin))
            # adjust to account for translation when rotated
            M[0, 2] += (rotated_width / 2) - center_x
            M[1, 2] += (rotated_height / 2) - center_y
            # rotate image 
            image = cv.warpAffine(image, M, (rotated_width, rotated_height))
            new_height, new_width = image.shape[:2]
            corners = AugmentBoundingBox.compute_corners(bboxes)
            # extract class c
            c = corners[:,8]
            corners = corners[:,:8].reshape(-1,2)

            ones = np.ones((corners.shape[0], 1))
            corners = np.hstack((corners, ones))
            # rotate corners
            corners = np.dot(M, corners.T).T
            corners = corners.reshape(-1,8)
            corners = np.hstack((corners, c[:, None]), dtype=np.float32)
            # convert corners to opencv format/pascal voc format
            bboxes = AugmentBoundingBox.corner_to_opencv(corners)
            scale_x = new_width / width
            scale_y = new_height / height
            image  = AugmentImage.resize(image, width)
            bboxes[:,:4] /= [scale_x, scale_y, scale_x, scale_y]
            bboxes = AugmentBoundingBox.clip(bboxes, img_size = height, min_visibility = 0.4)
        return image, bboxes

    def shuffle_channel(image , p = 0.1):
        if np.random.rand() <= p:
            b, g, r = cv.split(image)
            # shuffle the channels in a random order
            channel_order = np.random.permutation([0, 1, 2])
            # reorder the channels based on the random order
            image = cv.merge([b, g, r])[..., channel_order]
        return image

    def resize(image, resize_to):
        image = cv.resize(image, (resize_to, resize_to))
        return image
    
    def rgb_to_bgr(image):
        image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        return image
    
    def mosaic(image, bboxes,
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
               annotations_csv, img_dir, label_dir, p = 0.3):

        if np.random.rand() <= p:
            mosaic_images = []
            bboxes_lst = []
            bboxes_len = []
            annotations_csv = pd.read_csv(annotations_csv)
            mosaic_size = np.random.choice([4, 9])
            if mosaic_size == 4:
                mosaic_indices = random.sample(range(len(annotations_csv)), 4)

                for idx in mosaic_indices:
                    label_path = os.path.join(label_dir, annotations_csv.iloc[idx, 1])
                    img_path = os.path.join(img_dir, annotations_csv.iloc[idx, 0])
                    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)
                    image = np.array(Image.open(img_path).convert("RGB"))
                    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
                    
                    # convert to opencv pascal voc format
                    image = AugmentImage.resize(image, image_size // 2)
                    height, width = image.shape[:2]
                    bboxes = AugmentBoundingBox.yolo_to_opencv(bboxes, width, height)
                    
                    image, bboxes = AugmentImage.scale(image, bboxes, scale_factor, p = p_scale)
                    image, bboxes = AugmentImage.translate(image, bboxes, translate_factor, p = p_trans)
                    image, bboxes = AugmentImage.rotate(image, bboxes, rotation_angle, p = p_rot)
                    image, bboxes = AugmentImage.shear(image, bboxes, shear_angle, p = p_shear)
                    image, bboxes = AugmentImage.horizontal_flip(image, bboxes, p = p_hflip)
                    #image, bboxes = AugmentImage.vertical_flip()
                    #image, bboxes = AugmentImage.mixup()
                    #image, bboxes = AugmentImage.mosaic()
                    bboxes_len.append(len(bboxes))
                    image = AugmentImage.hsv_augment(image, hgain, sgain, vgain, p = p_hsv)
                    image = AugmentImage.greyscale(image, p = p_grey)
                    image = AugmentImage.blur(image, p = p_blur)
                    image = AugmentImage.clahe(image, p = p_clahe)
                    image = AugmentImage.cutout(image, p = p_cutout)
                    image = AugmentImage.shuffle_channel(image, p = p_shuffle)
                    image = AugmentImage.posterize(image, p = p_post)

                    bboxes_lst.extend(bboxes)
                    mosaic_images.append(image)
                image = np.vstack([np.hstack([mosaic_images[0], mosaic_images[1]]), np.hstack([mosaic_images[2], mosaic_images[3]])])
                bboxes_len = np.cumsum(bboxes_len).tolist()
                bboxes = np.array(bboxes_lst)

                # 1.top left image of mosaic: no adjustment so skip
                # note: height is the height of a single mosaic tile not the height of the total image
                # 2.top right image of mosaic: (x1,y1,x2,y2) + (height, 0, height, 0)
                bboxes[bboxes_len[0]:bboxes_len[1], :4] += [height , 0, height, 0]
                # 3. bottom left image of mosaic: (x1,y1,x2,y2) + (0, height, 0, height)
                bboxes[bboxes_len[1]:bboxes_len[2], :4] += [0, height, 0, height]
                # 4. bottom right image of mosaic: (x1,y1,x2,y2) + (height, height, height, height)
                bboxes[bboxes_len[2]:bboxes_len[3], :4] += [height, height, height, height]

            if mosaic_size == 9:
                mosaic_indices = random.sample(range(len(annotations_csv)), 9)

                for idx in mosaic_indices:
                    label_path = os.path.join(label_dir, annotations_csv.iloc[idx, 1])
                    img_path = os.path.join(img_dir, annotations_csv.iloc[idx, 0])
                    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)
                    image = np.array(Image.open(img_path).convert("RGB"))
                    image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
                    
                    # convert to opencv pascal voc format
                    image = AugmentImage.resize(image, image_size // 3)
                    height, width = image.shape[:2]
                    bboxes = AugmentBoundingBox.yolo_to_opencv(bboxes, width, height)

                    image, bboxes = AugmentImage.scale(image, bboxes, scale_factor, p = p_scale)
                    image, bboxes = AugmentImage.translate(image, bboxes, translate_factor, p = p_trans)
                    image, bboxes = AugmentImage.rotate(image, bboxes, rotation_angle, p = p_rot)
                    image, bboxes = AugmentImage.shear(image, bboxes, shear_angle, p = p_shear)
                    image, bboxes = AugmentImage.horizontal_flip(image, bboxes, p = p_hflip)
                    #image, bboxes = AugmentImage.vertical_flip()
                    #image, bboxes = AugmentImage.mixup()
                    #image, bboxes = AugmentImage.mosaic()
                    bboxes_len.append(len(bboxes))
                    image = AugmentImage.hsv_augment(image, hgain, sgain, vgain, p = p_hsv)
                    image = AugmentImage.greyscale(image, p = p_grey)
                    image = AugmentImage.blur(image, p = p_blur)
                    image = AugmentImage.clahe(image, p = p_clahe)
                    image = AugmentImage.cutout(image, p = p_cutout)
                    image = AugmentImage.shuffle_channel(image, p = p_shuffle)
                    image = AugmentImage.posterize(image, p = p_post)
                    
                    bboxes_lst.extend(bboxes)
                    mosaic_images.append(image)
                image = np.vstack([np.hstack([mosaic_images[0], mosaic_images[1], mosaic_images[2]]),
                                   np.hstack([mosaic_images[3], mosaic_images[4], mosaic_images[5]]),
                                   np.hstack([mosaic_images[6], mosaic_images[7], mosaic_images[8]])])
                
                bboxes_len = np.cumsum(bboxes_len).tolist()
                bboxes = np.array(bboxes_lst)
                # 1.top left image of mosaic: no adjustment so skip
                # note: height is the height of a single mosaic tile not the height of the total image
                # 2.top middle image of mosaic: (x1,y1,x2,y2) + (height, 0, height, 0)
                bboxes[bboxes_len[0]:bboxes_len[1], :4] += [height, 0, height, 0]
                # 3.top right image of mosaic: (x1,y1,x2,y2) + (height*2, 0, height*2)
                bboxes[bboxes_len[1]:bboxes_len[2], :4] += [height*2, 0, height*2, 0]
                # 4.middle left image of mosaic: (x1,y1,x2,y2) + (0, height, 0, height)
                bboxes[bboxes_len[2]:bboxes_len[3], :4] += [0, height, 0, height]
                # 5.middle middle image of mosaic: (x1,y1,x2,y2) + (height, height, height, height)
                bboxes[bboxes_len[3]:bboxes_len[4], :4] += [height, height, height, height]
                # 6.middle right image of mosaic: (x1,y1,x2,y2) + (height*2, height, height*2, height)
                bboxes[bboxes_len[4]:bboxes_len[5], :4] += [height*2, height, height*2, height]
                # 7.bottom left image of mosaic: (x1,y1,x2,y2) + (0, height*2, 0, height*2)
                bboxes[bboxes_len[5]:bboxes_len[6], :4] += [0, height*2, 0, height*2]
                # 8.bottom middle  image of mosaic: (x1,y1,x2,y2) + (height, height*2, height, height*2)
                bboxes[bboxes_len[6]:bboxes_len[7], :4] += [height, height*2, height, height*2]
                # 9.bottom right image of mosaic: (x1,y1,x2,y2) + (height*2, height*2, height*2, height*2)
                bboxes[bboxes_len[7]:bboxes_len[8], :4] += [height*2, height*2, height*2, height*2]
        return image, bboxes
    
    def mixup(image, bboxes,
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
               annotations_csv, img_dir, label_dir, p = 0.3):

        if np.random.rand() <= p:
            annotations_csv = pd.read_csv(annotations_csv)
            mixup_indices = random.sample(range(len(annotations_csv)), 2)
            mixup_images = []
            mixup_bboxes = []
            for idx in mixup_indices:
                label_path = os.path.join(label_dir, annotations_csv.iloc[idx, 1])
                img_path = os.path.join(img_dir, annotations_csv.iloc[idx, 0])
                bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)

                image = np.array(Image.open(img_path).convert("RGB"))
                image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
                # convert to opencv pascal voc format
                image = AugmentImage.resize(image, image_size)
                height, width = image.shape[:2]
                bboxes = AugmentBoundingBox.yolo_to_opencv(bboxes, width, height)

                image, bboxes = AugmentImage.scale(image, bboxes, scale_factor, p = p_scale)
                image, bboxes = AugmentImage.translate(image, bboxes, translate_factor, p = p_trans)
                image, bboxes = AugmentImage.rotate(image, bboxes, rotation_angle, p = p_rot)
                image, bboxes = AugmentImage.shear(image, bboxes, shear_angle, p = p_shear)
                image, bboxes = AugmentImage.horizontal_flip(image, bboxes, p = p_hflip)
                #image, bboxes = AugmentImage.vertical_flip()
                #image, bboxes = AugmentImage.mixup()
                #image, bboxes = AugmentImage.mosaic()
                image = AugmentImage.hsv_augment(image, hgain, sgain, vgain, p = p_hsv)
                image = AugmentImage.greyscale(image, p = p_grey)
                image = AugmentImage.blur(image, p = p_blur)
                image = AugmentImage.clahe(image, p = p_clahe)
                image = AugmentImage.cutout(image, p = p_cutout)
                image = AugmentImage.shuffle_channel(image, p = p_shuffle)
                image = AugmentImage.posterize(image, p = p_post)

                mixup_images.append(image)
                mixup_bboxes.extend(bboxes)
            lambda_mix = np.random.beta(8.0, 8.0)
            image = (mixup_images[0] * lambda_mix + mixup_images[1] * (1 - lambda_mix)).astype(np.uint8)
            bboxes = np.array(mixup_bboxes)
        return image, bboxes

class AugmentBoundingBox:
    @staticmethod
    def compute_area(bboxes):
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:,3] - bboxes[:,1])
        return area

    def clip(bboxes, img_size, min_visibility = 0.4, width_height_threshold = 4, aspect_ratio_threshold = 20):
        eps = 1e-9
        original_area = AugmentBoundingBox.compute_area(bboxes)
        # clips bounding boxes to be within the image
        # cliped below at 1 and above  at image -1  assuming 
        # a square image of size image size by image size 
        bboxes[:, :4] = np.clip(bboxes[:, :4], 1, img_size - 1, dtype=np.float32)

        new_area = AugmentBoundingBox.compute_area(bboxes)
        area_dif = (original_area - new_area) / (new_area + eps)

        # Compute width and height of bounding boxes
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        aspect_ratios = np.maximum(widths / (heights + eps), heights / (widths + eps))

        # Mask out bounding boxes based on width and height threshold, min visbibility and aspect ratio
        mask = ( (widths > width_height_threshold) & 
                    (heights > width_height_threshold) & 
                    (aspect_ratios < aspect_ratio_threshold) & 
                    (area_dif < (1 - min_visibility))
                    ).astype(int)

        bboxes = bboxes[mask == 1,:]

        return bboxes

    def compute_corners(bboxes):
        c = (bboxes[:,4])[:, None]
        # width: bottom corner right x2 - top left corner x1 
        width = (bboxes[:,2] - bboxes[:,0])[:, None]
        # height: bottom 
        height = (bboxes[:,3] - bboxes[:,1])[:, None]
        x1 = (bboxes[:,0])[:, None]
        y1 = (bboxes[:,1])[:, None]
        # compute top right corner values x2, y2
        x2 = x1 + width
        y2 = y1 
        # compute bottom left corner values x3, y3
        x3 = x1
        y3 = y1 + height
        x4 = (bboxes[:,2])[:, None]
        y4 = (bboxes[:,3])[:, None]
        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4,c))
        return corners
    
    def corner_to_opencv(corners):
        c = (corners[:,8])[:,None]
        x_corner_coords = corners[:,[0,2,4,6]] 
        y_corner_coords = corners[:,[1,3,5,7]]
        x1 = np.min(x_corner_coords,1)[:, None]
        y1 = np.min(y_corner_coords,1)[:, None]
        x2 = np.max(x_corner_coords,1)[:, None]
        y2 = np.max(y_corner_coords,1)[:, None]
        return np.hstack((x1, y1, x2, y2, c))
        
    def opencv_to_yolo(bboxes, width, height):
        """
        Convert pascal voc or opencv bounding box coordinates to YOLO format.
        Opencv format are two points: 
            - point 1 = (topleft corner x, topleft corner y) = x1, y1
            - point 2 = (bottom right corner x, bottom right corner y) = x2, y2

        Arguments:
        - boxes: np.array of shape N,4: N number of boxes and 4 = point1(x1, y1), point2(x2, y2)
        - width: Width of the image.
        - height: Height of the image.

        Returns:
        Yolo format bounding box centerx, centery, width, height of bounding box.
        """
        if bboxes.ndim == 1:
            #print("Empty bounding box:", bboxes)
            # if no bboxes then return bboxes and handle exception in dataset.py
            return bboxes
        c = bboxes[:,4]
        x1, y1, x2, y2 = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
        x = (x2 + x1) / (2 * width)
        y = (y2 + y1) / (2 * height)
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        # return yolo format: xcenter, ycenter, width, height normalised
        return np.column_stack((x, y, w, h, c))

    def yolo_to_opencv(bboxes, width, height):
        """
        Convert YOLO-format bounding box coordinates to OpenCV-format.

        Arguments:
        - boxes: np.array of shape N,4: N number of boxes and 4 x, y, w, h Yolo format
        - width: Width of the image in pixels.
        - height: Height of the image in pixels.

        Returns:
        x1, y1, x2, y2: Coordinates of the top-right and bottom-left corners of the bounding box in OpenCV format.
        """
        c = bboxes[:,4]
        x, y, w, h = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
        w = w * width
        h = h * height
        x1 = ((2 * x * width) - w) / 2
        y1 = ((2 * y * height) - h) / 2
        x2 = x1 + w
        y2 = y1 + h
        return np.column_stack((x1, y1, x2, y2, c))
