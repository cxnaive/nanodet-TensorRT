import numpy as np


def merge_box(box1, box2):
    xl = max(box1[0],box2[0])
    xr = min(box1[2],box2[2])
    yl = max(box1[1],box2[1])
    yr = min(box1[3],box2[3])
    if xl >= xr or yl >= yr:
        return None
    else:
        area_before = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_after = (xr - xl) * (yr - yl)
        
        #eliminate incomplete boxes 
        if area_after / area_before < 0.2 or (xr - xl) / (box1[2] - box1[0]) < 0.4 or (yr - yl) / (box1[3] - box1[1]) < 0.4:
            return None
        return [xl,yl,xr,yr]


def merge_bboxes(bboxes, labels, cutx, cuty, im_w, im_h):
    merge_bbox = []
    merge_label = []
    for i in range(len(bboxes)):
        for j in range(len(bboxes[i])):
            # print(labels[i][j], bboxes[i][j])
            if i == 0:
                box_now = merge_box(bboxes[i][j], [0, 0, cutx, cuty])
            if i == 1:
                box_now = merge_box(bboxes[i][j], [0, cuty, cutx, im_h])
            if i == 2:
                box_now = merge_box(bboxes[i][j], [cutx, cuty, im_w, im_h])
            if i == 3:
                box_now = merge_box(bboxes[i][j], [cutx, 0, im_w, cuty])
    
            if box_now == None:
                continue
            
            merge_bbox.append(box_now)
            merge_label.append(labels[i][j])
    # print(merge_bbox)
    if len(merge_bbox) == 0:
        return [], []
    else:
        return merge_label, merge_bbox
