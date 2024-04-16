# Another important method for object detection is Non-Max-Suppression. 
# For one object, we can have multiple bounding box predicitons.
# Thus, the question - how do we get the best result from all these predictions. 
# And thats where Non-Max-Suppression comes into play. 

import torch
from IoU import intersection_over_union

def non_max_suppression(bboxes, iou_threshold, prob_threshold, box_format='corners'):
    # predictions = [class, probability score, x1,y1,x2,y2]
    
    assert type(bboxes) == list
    
    # Keep if probability threshold is valid
    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    # Sort to have highest probability in beginning
    bboxes = sorted(bboxes, key= lambda x: x[1], reverse = True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        
        # Compare box to chosen box with regards to IoU
        bboxes = [box 
                  for box in bboxes 
                  if box[0] != chosen_box[0]
                  or intersection_over_union(
                      # remove class and probability score
                      torch.tensor(chosen_box[2:]),
                      torch.tensor(box[2:])
                  ) < iou_threshold]
        
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms
    
    