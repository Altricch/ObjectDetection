# Mean Average Precision 
# Foe explanation look into overleaf document

import torch
from collections import Counter
from IoU import intersection_over_union

def mean_average_precision(pred_boxes, true_boxes, iou_threshold = 0.5, num_classes = 20):
    
    average_precision = []
    epsilon = 1e-6

    # pred_boxes = [train_idx (which image of train), class_pred, prob_score, x1, y1, x2, y2]
    for cls in range(num_classes):
        detections = []
        groundtruth = []
        
        # how many and which bboxes were predicted per class
        for detection in pred_boxes:
            if detection[1] == cls:
                detections.append(detection)
        
        # how many and which bboxes / objects are actaully in image per class
        for true_box in true_boxes:
            if true_box[1] == cls:
                groundtruth.append(true_box)
        
        # img 0 has 3 bboxes
        # img 1 has 5 bboxes
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in groundtruth])
        
        # Need to keep track of the target bounding boxes that we have covered so far
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        # amount_bboxes = {0: torch.tensor([0,0,0], ...)}    
        
        detections.sort(key=lambda x: x[2], reverse = True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(groundtruth)
        
        # Extracted particular bounding box for a particular class
        for detection_idx, detection in enumerate(detections):
            # Just compare them for the same image
            groundtruth_img = [bbox for bbox in groundtruth if bbox[0] == detection[0]]
            num_gts = len(groundtruth_img)
            best_iou = 0
            # best_gt_idx = -1
            
            # Go through all the bounding boxes in the image
            # and extract best IoU for a gt given a detection bbox
            for idx, gt in enumerate(groundtruth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]),
                                              torch.tensor(gt[3:]))
                # Store idx of best gt index
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx
            
            # If best is better than threhold, that means prediction is correct
            if best_iou > iou_threshold:
                # if number of bounding box for #img (detection[0]) and get 
                # target bounding box index is == 0, means that this target bounding box
                # has not been covered
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    # Else box was already covered
                    FP[detection_idx] = 1
            
            # There was no IoU over this threshold
            else:
                FP[detection_idx] = 1
        
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precision = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        # Add one to the precisions for numerical integration to x axis
        precision = torch.cat([torch.tensor([1]), precision])
        recalls = torch.cat([torch.tensor([0]), recalls])
        
        # Get are under precision recalls   
        average_precision.append(torch.trapz(precision, recalls))
        
    return sum(average_precision) / len(average_precision)
        
        