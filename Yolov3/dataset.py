# import config
import torch 
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

from torch.utils.data import Dataset, DataLoader

from utils import (iou_width_height as iou,
                   non_max_suppression as nms)


ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLODataset(Dataset):
    def __init__(self, csv, img_dir, label_dir, anchors, image_size=416,S=[13,26,52], C=20, transform=None):
        self.annotations = pd.read_csv(csv)
        self.img_dir = img_dir
        self.label_dir = label_dir
        
        
        self.transform = transform
        self.S = S
        
        # Put all anchors of all three scales together
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        # index, 1 is where name is located
        print("LABEL DIR", self.label_dir)
        print("ANNOTATION", self.annotations.iloc[index, 1])
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        print("LABEL PATH",label_path)
        #[class, x,y,w,h] as per text file, to augment we use albumentation, they want [x,y,w,h, class], thus, np.roll
        bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).tolist() 
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.transform:
            augmentations = self.transform(image = image, bboxes = bboxes)
            image = augmentations['image']
            bboxes = augmentations['bboxes']
        
        # For each cell for each grid size, 3 anchors, S , S, 6 bc. [p, x, y, w, h, c]
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S] 

        
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim = 0)
            x, y, width, height, class_label = box
            # Must make sure that there is a bounding box for each of the anchors predicting the object
            has_anchor = [False, False, False]
            
            for anchor_idx in anchor_indices:
                # We check which scale index it belongs to [13,26,52]
                scale_idx = anchor_idx // self.num_anchors_per_scale # output [0,1,2] is index to take out from target
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale # output 0,1,2 which anchor in that particular scale
                
                S = self.S[scale_idx]
                i, j = int(S*y), int(S * x) # e.g. x = 0.5, S = 13 --> int(6.5) = 6
                # Take out 0 for proability that there is an object
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                
                # make sure anchor has not been taken and we dont have an anchor on this
                # particular scale for this bounding box
                if not anchor_taken and not has_anchor[scale_idx]: 
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    # Get cell coordinates
                    x_cell, y_cell = S*x-j, S*y - i
                    width_cell, height_cell = (
                        width*S, 
                        height*S
                    )
                    
                    box_coordinates = torch.tensor(
                        [x_cell, y_cell, width_cell, height_cell]
                    )
                    
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                    
                    
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i,j,0] = -1 #ignore index
                    
        return targets, image  
