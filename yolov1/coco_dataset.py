import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from pycocotools.coco import COCO

from PIL import Image
from typing import Any, Callable, Optional, Tuple, List
import os


class COCODataset(Dataset):
    def __init__(self, 
                 is_train: str, 
                 data_type: str, 
                 image_size: int = 448, 
                 num_grid: int = 7, 
                 num_bboxes: int = 2, 
                 num_classes: int = 80,
                 transforms: List[Callable] = None) -> None:
        """ Constructor
        Args:
            is_train (str): [description]
            data_type (str): [description]
            image_size (int, optional): [description]. Defaults to 448.
            num_grid (int, optional): [description]. Defaults to 7.
            num_bboxes (int, optional): [description]. Defaults to 2.
            num_classes (int, optional): [description]. Defaults to 80.
            transforms (List[Callable], optional): [description]. Defaults to None.
        """
        
        self.is_train = is_train
        self.image_size = image_size
        
        self.S = num_grid
        self.B = num_bboxes
        self.C = num_classes
    
        self.data_dir = "coco"
        self.data_type = data_type
        ann_path ='{}/annotations/instances_{}.json'.format(self.data_dir, self.data_type)
        self.coco = COCO(ann_path)
        self.ids = list(self.coco.imgs.keys())
        
        self.transforms = transforms
        
    def __getitem__(self, index: int) -> Tuple:
        """
        Args:
            index (int): index of image and annotation to process and return
        Returns:
            tuple: image, target tensor and annotation 
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        boxes = [ann["bbox"] for ann in anns]  # [[x, y, width, height],...]
        labels = [ann["category_id"] for ann in anns]
        
        img_path = coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.data_dir, self.data_type, img_path))
        
        # Transformations should happen her
        if self.transforms is not None:
            for t in self.transforms:
                img, boxes = t(img, boxes)
        
        boxes = torch.from_numpy(np.array(boxes))
        boxes = self.center_boxes(boxes)
        boxes = self.normalize_boxes(boxes, img.size)
        print(boxes)
        
        # Need to resize the picture first
        targets = self.encode(boxes, labels)
        #img = img / 255.0  # Simple normalization, map [0, 255] -> [0, 1]
        return img, targets, anns
    
    def center_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """ Move upper left corner to middle of bbox and half w and h
        Args:
            boxes (tensor): tensor of bboxes on the COCO format, size [n_bboxes, 4]
        Returns:
            tensor: tensor of bboxes where xy represent the center of the bboxes
        """
        boxes[:, 2:] *= 0.5
        boxes[:, 0] += boxes[:, 2]
        boxes[:, 1] += boxes[:, 3]
        return boxes
    
    def normalize_boxes(self, boxes, image_size) -> torch.Tensor:
        """ Normalize bboxes from 0.0 to 1.0 w.r.t. image size
        Args:
            boxes (tensor): tensor of centered bboxes
            image_size (tuple): width and hight of current image
        Returns:
            tensor: tensor of bboxes normalized w.r.t. image size
        """
        w, h = image_size[0], image_size[1]
        boxes /= torch.Tensor([[w, h, w, h]]).expand_as(boxes)
        return boxes
    
    def encode(self, boxes: torch.Tensor, labels: List[int]) -> torch.Tensor:
        """ Encode bbox coordinates and class labels as one target tensor
        Args:
            boxes (tensor): tensor of centered bboxes. Normalized w.r.t. image widht/height, size [n_bboxes, 4]
            lables (list of int): list of category id, size [n_bboxes]
        Returns:
            tensor: An encoded representation of bboxes, size [S, S, B * 5 + C], 5=(x, y, w, h, conf)
        """
        N = self.B * 5 + self.C
        targets = torch.zeros(self.S, self.S, N)
        
        cell_size = 1.0 / float(self.S)  # We use 1.0 as the boxes has been normalize w.r.t. image shape
        
        # TODO: make it possible for two bboxes to be in the same grid cell without one of them beeing removed from the target tensor
        for b in range(len(boxes)):
            xy = boxes[b, :2]
            wh = boxes[b, 2:]
            label = labels[b]
            
            # Find grid cell the center of the bbox belongs to
            ij = (xy / cell_size).floor()
            i, j = int(ij[0]), int(ij[1])
            
            # Normalize the bbox to the size of the grid cell
            c_xy =  ij * cell_size # Top left corner of cell
            xy_normalized = (xy - c_xy) / cell_size
            
            # Assign the targets tensor values for bbox
            for k in range(self.B):
                s = 5 * k
                targets[i, j, s:s+2] = xy_normalized
                targets[i, j, s+2:s+4] = wh
                targets[i, j, s+4] = 1.0
            targets[i, j, 5 * self.B + label] = 1.0
            
        return targets