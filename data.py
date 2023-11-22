import cv2, torch
import numpy as np
import torch.nn as nn

from utils import bbox2yolo, yolo2bbox
from PIL import Image
from torchvision.ops import box_iou



class YOLODataset(nn.Module):
    def __init__(self, data, encoder, anchors, transform=None):
        super(YOLODataset, self).__init__()
        self.data = data
        self.encoder = encoder
        self.transform = transform
        self.anchors = anchors
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, x):
        # image = np.array(Image.open(self.data[x][0]).convert("RGB"))
        image = cv2.imread(self.data[x][0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (320, 320), interpolation=cv2.INTER_CUBIC)
        
        
        bboxes = self.data[x][1]
        
        Wratio = resized_image.shape[1]/image.shape[1]
        Hratio = resized_image.shape[0]/image.shape[0]

        ratioList = [Wratio, Hratio, Wratio, Hratio]
        resized_bboxs, classes = [], []
        for row in bboxes:
            q = []            
            cls, xmin, ymin, xmax, ymax = row
            bbox = [int(a * b) / 320 for a, b in zip([xmin, ymin, xmax, ymax], ratioList)]
            classes.append(cls)
            bbox = bbox2yolo(bbox)
            resized_bboxs.append(bbox)
        bboxes = torch.tensor(resized_bboxs)
        # bboxes = torch.clip(bboxes, 0, 1)
        # print(bboxes, "before")
        # ----------------------------------------------------------------------------------------
        
        if self.transform:
            augmentations = self.transform(image=resized_image, bboxes=bboxes)
            resized_image = augmentations["image"]
            bboxes = torch.tensor(augmentations["bboxes"])
        
        # print(bboxes, "after")
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((2, S, S, 8)) for S in [40, 20]]
        
        for cls, bbox in zip(classes, bboxes):
            iou_res = box_iou(bbox[None], self.anchors)[0]
            anchor_indexs = iou_res.argsort(descending=True, dim=-1)
            has_anchor = [False] * 2 # each scale should have one anchor
            # xmin, ymin, xmax, ymax = box_corner_to_center(bbox) # corner_to_center(bbox)
            xmin, ymin, xmax, ymax = bbox
            
            for anchor_idx in anchor_indexs:
                i, j = int(40 * xmin), int(40 * ymin)
                w_cell, h_cell = xmax * 40, ymax * 40

                anchor_taken = targets[0][anchor_idx, i, j, 0]
                if not anchor_taken and not has_anchor[0]:
                    targets[0][anchor_idx, i, j, 0] = 1
                    targets[0][anchor_idx, i, j, 1:5] = torch.tensor([xmin, ymin, w_cell, h_cell])
                    targets[0][anchor_idx, i, j, 5:] = self.encoder[cls]
                    has_anchor[0] = True
                elif not anchor_taken and iou_res[anchor_idx] > 0.5:
                    targets[0][anchor_idx, i, j, 0] = -1  # ignore prediction

                # ---------------------------------------------- 

                i, j = int(20 * xmin), int(20 * ymin)
                w_cell, h_cell = xmax * 20, ymax * 20
                anchor_taken = targets[1][anchor_idx, i, j, 0]
                if not anchor_taken and not has_anchor[1]:
                    targets[1][anchor_idx, i, j, 0] = 1
                    targets[1][anchor_idx, i, j, 1:5] = torch.tensor([xmin, ymin, w_cell, h_cell])
                    targets[1][anchor_idx, i, j, 5:] = self.encoder[cls]
                    has_anchor[1] = True
                elif not anchor_taken and iou_res[anchor_idx] > 0.5:
                    targets[1][anchor_idx, i, j, 0] = -1  # ignore prediction
        
        return resized_image / 255, targets
