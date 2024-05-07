import torch 

def intersection_over_union(boxes_pred, boxes_label):
    # ... are all dimensions before, 
    # boxes.shape is (N, 4) w/ N number of bounding boxes
    # Must do slixing to preserve second dimension
    box1_x1 = boxes_pred[..., 0:1]
    box1_y1 = boxes_pred[..., 1:2]
    box1_x2 = boxes_pred[..., 2:3]
    box1_y2 = boxes_pred[..., 3:4]
    
    box2_x1 = boxes_label[..., 0:1]
    box2_y1 = boxes_label[..., 1:2]
    box2_x2 = boxes_label[..., 2:3]
    box2_y2 = boxes_label[..., 3:4]
    
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.max(box1_x2, box2_x2)
    y2 = torch.max(box1_y2, box2_y2)
    
    # Clamp for if they dont intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # Union prep
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    
    # +1e-6 for numerical stability
    return intersection / (box1_area + box2_area - intersection + 1e-6)
    