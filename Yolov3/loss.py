import torch
import torch.nn as nn

from utils import intersection_over_union



class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

        # For box loss
        self.mse = nn.MSELoss()
        
        self.bce = nn.BCEWithLogitsLoss()
        
        # Class loss (paper uses BCE, but we inly have one label per box, so we use CrossEntropu)
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        
        # Constants
        self.labmda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
    
    # Compute loss for every single scale. We have to do it in train for 
    # every scale
    def forward(self, pred, target, anchors):
        
        # Find in targets where there is and isnt an object
        obj = target[..., 0] == 1
        noobj = target[..., 0] == 0
        
        ####### LOSS FOR NO OBJECT #######
        no_obj_loss = self.bce(
            (pred[..., 0:1][noobj]), (target[..., 0:1][noobj])
        )
        
        ####### LOSS FOR OBJECT #######
        # reshape to match. Anchors are 3x2 (each anchor has h,w) 
        anchors = anchors.reshape(1,3,1,1,2)
        box_pred = torch.cat([self.sigmoid(pred[..., 1:3]), torch.exp(pred[..., 3:5]) * anchors], dim = -1)
        ious = intersection_over_union(box_pred[obj], target[..., 1:5][obj]).detach()
        
        object_loss = self.mse(self.sigmoid(pred[..., 0:1][obj]), ious * target[..., 0:1][obj])
        
        
        
        
        ####### LOSS FOR BOX COORDINATE #######
        pred[..., 1:3] = self.sigmoid(pred[..., 1:3]) # x & y to be between [0,1]
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )
        
        box_loss = self.mse(pred[..., 1:5][obj], target[..., 1:5][obj])
        
        ####### CLASS LOSS #######
        class_loss = self.entropy(
            (pred[..., 5:][obj]), (target[..., 5][obj].long()),
        )
        
        return (
            self.lambda_box * box_loss
            + self.lambda_obj * object_loss
            + self.lambda_noobj + no_obj_loss
            + self.labmda_class + class_loss
        )
        
        
        
        