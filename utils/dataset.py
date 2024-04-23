import torch 
import os
import cv2 as cv
import pandas as pd
from PIL import Image, ImageFile
from utils.utils import iou_width_height
from utils.utils import non_max_suppression
from utils.utils import cells_to_bboxes
from utils.utils import draw_bounding_box
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.transforms import augment_data
import warnings

image_size = 608

if image_size == 608:
    anchors = [
  [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
  [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
  [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
  ]
    S = [19, 38, 76]

if image_size == 416:
    anchors = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
    S = [13, 26, 52]

class CoCoDataset(Dataset):
    def __init__(self, csv_file, img_dir, label_dir, anchors, image_size = image_size, 
                 S = S, C = 80, mode = 'test'):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.mode = mode
        self.S = S
        # anchor for 3 scales
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])  
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(fname = label_path, delimiter = " ", ndmin = 2), 4, axis = 1).astype(np.float32)
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.mode == 'test':
            image, bboxes = augment_data(image, bboxes, image_size = image_size, mode = self.mode)
        
        if self.mode == 'train':
            image, bboxes = augment_data(image, bboxes, image_size = image_size,
                                         p_scale = 1.0,  scale_factor = 0.9,
                                         p_trans = 1.0, translate_factor = 0.1,
                                         p_rot = 0.3, rotation_angle = 45.0,
                                         p_shear = 0.0, shear_angle = 10.0,
                                         p_hflip = 0.5,
                                         p_vflip = 0.0,
                                         p_mixup = 0.3,
                                         p_mosaic = 0.3,
                                         p_hsv = 1.0, hgain = 0.015, sgain = 0.7, vgain = 0.4,
                                         p_grey = 0.1,
                                         p_blur = 0.1,
                                         p_clahe = 0.1,
                                         p_cutout = 0.0,
                                         p_shuffle = 0.1,
                                         p_post = 0.1, mode = self.mode)
	
        #print("Dataset test boxes:", bboxes)
        #print("dataset test boxes shape:", bboxes.shape)
        image = np.array(image)
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
        
        # if bboxes is empty, return black image and targets
        if bboxes.ndim == 1:
            #print("Empty bounding box:", bboxes)
            #print("Targets:", tuple(targets))
            image = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)
            return image, tuple(targets)

        for box in bboxes:

            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending = True, dim = 0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3  # each scale should have one anchor
            
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx] 
                i, j = int(S * y), int(S * x)  # which cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                    width_cell, height_cell = (
                        width * S,
                        height * S,
                    )  # can be greater than 1 since it's relative to cell
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
        #print("Dataset targets:", tuple(targets))
        return image, tuple(targets)

def test(anchors = anchors, mode = 'train', image_size = image_size):

    dataset = CoCoDataset("data/coco/train_10examples.csv", "data/coco/images/", "data/coco/labels/",
                          S = S, anchors = anchors, image_size = image_size,  mode = mode)
    
    scaled_anchors = torch.tensor(anchors) / ( 1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) )
    loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = False)
    
    for idx, (x, y) in enumerate(loader):
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(y[i], is_preds=False, S=y[i].shape[2], anchors = anchor)[0]
        
        boxes = non_max_suppression(boxes, iou_threshold = 1.0, threshold = 0.7, mode = "ciou")
        #print(boxes)
        #print(x[0].shape)
        print('x[0] shape:', x[0].shape)
        print('x shape:', x.shape)
        print('x[0].permute(0, 1, 2):', x[0].permute(0, 1, 2).shape)
        img = draw_bounding_box(x[0].permute(0, 1, 2).to("cpu") * 255, boxes)
        filename = f'figures/yolo_data_{idx}.png' 
        plt.imsave(filename, img)

#test()


