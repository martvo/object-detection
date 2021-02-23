import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov1.coco_dataset import COCODataset


class YOLOv1Loss(nn.Module):
    def __init__(self, 
                 num_grid: int = 7, 
                 num_bboxes: int = 2, 
                 num_classes: int = 80,
                 lambda_coord: int = 5.0,
                 lambda_noobj: int = 0.5
                 ) -> None:
        """ Constructor
        Args:
            num_grid: (int)
            num_bboxes: (int)
            num_classes: (int)
            lambda_coord: (float)
            lambda_noobj: (float)
        """
        super(YOLOv1Loss, self).__init__()
        
        self.S = num_grid
        self.B = num_bboxes
        self.C = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        
    def IoU(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> float:
        """ Compute the Intersection over Union
        Args:
            bboxes1 (tensor): bounding boxes on format [x1, y1, x2, y2], size [N, 4]
            bboxes2 (tensor): bounding boxes on format [x1, y1, x2, y2], size [M, 4]
        Returns:
            iou (tensor): Intersection over Union between bboxes1 and bboxes2, size [N, M]
        """
        N = bboxes1.size(0)
        M = bboxes2.size(0)
        
        # Top left corner of intersection
        tl = torch.max(
            bboxes1[:, :2].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bboxes2[:, :2].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        
        # Bottom right corner of intersection
        br = torch.min(
            bboxes1[:, 2:].unsqueeze(1).expand(N, M, 2), # [N, 2] -> [N, 1, 2] -> [N, M, 2]
            bboxes2[:, 2:].unsqueeze(0).expand(N, M, 2)  # [M, 2] -> [1, M, 2] -> [N, M, 2]
        )
        
        wh = br - tl
        print("wh", wh)
        wh[wh < 0] = 0
        intersection = wh[:, :, 0] * wh[:, :, 1]

        bboxes1_area = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        bboxes2_area = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        bboxes1_area = bboxes1_area.unsqueeze(1).expand_as(intersection)
        bboxes2_area = bboxes2_area.unsqueeze(0).expand_as(intersection)

        union = bboxes1_area + bboxes2_area - intersection
        iou = intersection / union
        return iou
    
    def GIoU(self, bboxes1: torch.Tensor, bboxes2: torch.Tensor) -> float:
        """ Compute the Generalized Intersection over Union
        https://arxiv.org/pdf/1902.09630.pdf
        Args:
            bboxes1 (tensor): bounding boxes, size [batch_size, 4]
            bboxes2 (tensor): bounding boxes, size [batch_size, 4]
        Returns:
            giou (tensor): Generalized Intersection over Union between bboxes1 and bboxes2, size [batch_size, 1]
        """
        iou = IoU(bboxes1, bboxes2)
        giou = 0.0
        return giou
        
    def forward(self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> torch.float64:
        """ Compute loss based on https://arxiv.org/pdf/1506.02640.pdf
        Args:
            pred_tensor (tensor): model predictions, bboxes on yolo form [x, y, w, h], size [batch_size, S, S, B * 5 + C]
            target_tensor (tensor): targets, size [batch_size, S, S, B * 5 + C]
        """
        batch_size = pred_tensor.size(0)
        N = 5 * self.B + self.C
        
        # get mask for coord and noobj grid cells
        noobj_mask = target_tensor[:, :, :, 4] == 0
        noobj_mask = noobj_mask.unsqueeze(-1).expand_as(target_tensor)
        coord_mask = target_tensor[:, :, :, 4] > 0.0
        coord_mask = coord_mask.unsqueeze(-1).expand_as(target_tensor)
        
        # Compute noobj confidence loss for grid cells not containing objects
        noobj_pred = pred_tensor.masked_select(noobj_mask).view(-1, N)
        noobj_target = target_tensor.masked_select(noobj_mask).view(-1, N)
        conf_columns = [4 + 5*b for b in range(self.B)]
        noobj_conf_pred = noobj_pred[:, conf_columns]
        noobj_conf_target = noobj_target[:, conf_columns]        
        noobj_conf_loss = F.mse_loss(noobj_conf_pred, noobj_conf_target, reduction="sum")
        
        # Compute loss for coord grid cells containing objects
        # TODO: Denormalize the 
        coord_pred = pred_tensor.masked_select(coord_mask).view(-1, N)
        coord_target = target_tensor.masked_select(coord_mask).view(-1, N)
        
        # TODO: Pick predicted bbox of the B bboxes that has the highest IoU with the target
        bboxes_pred = coord_pred[:, :5*self.B].contiguous().view(-1, 5)
        bboxes_target = coord_target[:, :5*self.B].contiguous().view(-1, 5)
        
        bboxes_pred_mask = torch.BoolTensor(bboxes_pred.size()).fill_(False)
        target_iou = torch.zeros(bboxes_target.size(0))
        
        for i in range(0, bboxes_target.size(0), self.B):
            pred = bboxes_pred[i:i+self.B]  # predicted bboxes in gird cell i
            print("pred", pred)
            pred_xyxy = torch.as_tensor(pred)  # size [B, 5]
            print("pred_xyxy", pred_xyxy)
            pred_xyxy[:, :2] = pred[:, :2] * (1 / float(self.S)) - 0.5 * pred[:, 2:4]
            print("pred_xyxy", pred_xyxy)
            pred_xyxy[:, 2:4] = pred[:, :2] * (1 / float(self.S)) + 0.5 * pred[:, 2:4]
            print("pred_xyxy", pred_xyxy)
            
            target = bboxes_target[i].unsqueeze(0)
            target_xyxy = torch.as_tensor(target)  # size [1, 5]
            target_xyxy[:, :2] = target[:, :2] * (1 / float(self.S)) - 0.5 * target[:, 2:4]
            target_xyxy[:, 2:4] = target[:, :2] * (1 / float(self.S)) + 0.5 * target[:, 2:4]
            
            print(pred_xyxy[:, :4])
            print(target_xyxy[:, :4])
            iou = self.IoU(pred_xyxy[:, :4], target_xyxy[:, :4])
            print(iou)
            print()
            max_iou, max_index = iou.max(0)
            bboxes_pred_mask[i + max_index] = True
            
            target_iou[i + max_index] = max_iou.data
            
        # bboxes with the highest IoU with the target bboxes
        bboxes_pred = bboxes_pred[bboxes_pred_mask].view(-1, 5)
        bboxes_target = bboxes_target[bboxes_pred_mask].view(-1, 5)
        target_iou = target_iou[bboxes_pred_mask[:, 0]] #.view(-1)
        print(bboxes_pred, bboxes_target, target_iou)
        
        # Calculate localization loss
        xy_loss = F.mse_loss(bboxes_pred[:, :2], bboxes_target[:, :2], reduction="sum")
        wh_loss = F.mse_loss(bboxes_pred[:, 2:4], bboxes_target[:, 2:4], reduction="sum")
        
        # Calculate coord confidence loss
        # we want the confidence score to equal the intersection over union (IOU) between the predicted box and the ground truth
        coord_conf_loss = F.mse_loss(bboxes_pred[:, 4], target_iou, reduction="sum")
        
        # Calculate classification loss
        class_pred = coord_pred[:, 5*self.B:]
        class_target = coord_target[:, 5*self.B:]
        class_loss = F.mse_loss(class_pred, class_target, reduction="sum")
        
        loss = self.lambda_coord * (xy_loss + wh_loss) + coord_conf_loss + self.lambda_noobj * noobj_conf_loss + class_loss
        return loss / float(batch_size)


if __name__ == "__main__":
    S, B, C = 7, 2, 80

    target = torch.zeros(S, B, C)
    prediction = torch.zeros(S, B, C)

    # Add link to a picture on cloudapp, put it a folder so that I wont deleat it
    # Target and pred tensor defined by the picture above

    print(target.shape, prediction.shape)
    dataset = COCODataset()
    criterion = YOLOv1Loss()