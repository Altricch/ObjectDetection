# Loss
# We take loss for predicted center vs actual center
# We also have a loss for each identity bounding box based on IoU
# The one that has highest IoU is then assigned identity for target
# We have two general losses, if there is an object or if there is no object in a certain cell

import torch
import torch.nn as nn
from utils import intersection_over_union



class YoloLoss(nn.Module):
    
    # Split size, boxes, classes
    # S = split size, B = number of boxes for each cell, C = number of classes
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        # They dont average here, thus we use sum
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        
    def forward(self, predictions, target):
        # First, reshape so we have (S,S,(C+B*5)), e.g. (S,S,30) for 2 bounding boxes
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B*5)
        
        # Calculate IoU for predicted BBoxes with target boxes
        # 21 ... 25 -> four bounding box values for the first bbox
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        # 26 ... 30 -> four bounding box values for the second bbox,but only one target so stay the same
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        # Return max values of the ious, and returns argmax (bestbox), thus which IoU is the best one
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # 20 is 0 or 1 if there is an object in that cell or no
        exists_box = target[...,20].unsqueeze(3) # identity of object i, is there an object in cell i
        
        # LOSS FOR BOX COORDINATES (midpoint, width and height)
        ###########################################
        
        
        # Only compute the loss if there is actually a box in cell
        box_predictions = exists_box * (
            (   
                # if best box is first one
                bestbox * predictions[..., 26:30] +
                # if second box was best one
                (1 - bestbox) * predictions[..., 21:25]
            )
        )
        
        
        
        box_targets = exists_box * target[...,21:25]
        
        # + 1e-6 for numerical stability, if its 0 (towards beginning), then goes against infinity the derivative
        
        # 2:4 is for width and height
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[...,2:4] + 1e-6))
        
        # (N, S, S, 25) -> ... for all N, S, S
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # (N, S, S, 4) -> (N*S*S, 4) thats how MSE expects the input 
        box_loss = self.mse(torch.flatten(box_predictions, end_dim=-2), 
                            torch.flatten(box_targets, end_dim=-2))
    
        
        # LOSS For object loss (if there is an object)
        ###########################################
        
        # take out probabiloty score, slicing to keep dimension
        pred_box = (bestbox * predictions[..., 25:26] + (1- bestbox) * predictions[..., 20:21])
        
        # (N*S*S)
        object_loss = self.mse(torch.flatten(exists_box * pred_box), 
                               torch.flatten(exists_box)* target[..., 20:21])
        
        
        # LOSS For no object loss (if there is no object)
        ###########################################
        
        # (N,S,S,1) -> (N, S*S)
        no_object_loss = self.mse(torch.flatten((1-exists_box) * predictions[..., 20:21], start_dim=1),
                                  torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        )
        
        
        no_object_loss += self.mse(torch.flatten((1-exists_box) * predictions[..., 25:26], start_dim=1),
                                  torch.flatten((1-exists_box) * target[..., 20:21], start_dim=1)
        )
        
        # LOSS For class loss (which class the object belongs to)
        ###########################################
        # (N,S,S,20) -> (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20],end_dim=-2)
        )
        
        loss = (
            self.lambda_coord * box_loss #first two rows of the paper
            + object_loss # third row of paper
            + self.lambda_noobj * no_object_loss # fourth row of paper
            + class_loss # fith row of paper
        )
        
        return loss
        
        
        
        
        
        