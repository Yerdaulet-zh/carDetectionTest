import torch
import torch.nn as nn
from utils import intersection_over_union


class YOLOLoss(nn.Module):
    def __init__(self):
        super(YOLOLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()
        
        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 1
        self.lambda_obj = 1
        self.lambda_box = 1
        
        
    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i
        
        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )
        
        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 2, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        
#         ious = []
#         with torch.no_grad():
#             for bbox, targ in zip(box_preds, target[..., 1:5][obj]):
#                 q = []
#                 q.append(box_iou(bbox[None], targ[None]).item())
#                 ious.append(q)
#         ious = torch.tensor(ious)
        
        # print(box_preds.shape)
        ious = intersection_over_union(box_preds[obj].reshape(-1, 4), target[..., 1:5][obj], box_format='midpoint').detach()
        # print(ious)
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])
        
        
        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # with torch.no_grad():
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])
        
        
        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )
        # print(box_loss, object_loss, no_object_loss, class_loss)
        
        return (
            self.lambda_box  * box_loss
            + self.lambda_obj * object_loss
            +self.lambda_noobj * no_object_loss
            + self.lambda_class * class_loss
        )