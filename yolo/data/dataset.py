"""Dataset Module"""

import os
import torch
from glob import glob
from PIL import Image
from typing import List, Tuple
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class Yolov1Dataset(Dataset):

    def __init__(self, annotation_dir, num_classes, num_grids, height=448, width=448):
        self.annot_dir = (
            annotation_dir if annotation_dir[-1] == "/" else annotation_dir + "/"
        )
        self.annotations = self._create_annotations()
        self.C = num_classes
        self.S = num_grids
        self.H = height
        self.W = width
        self.tt = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize((width, height))]
        )

    def _create_annotations(self) -> List[Tuple[str]]:
        """The method creates and checks sanity of the annotations done

        The annot_dir is expected to have the following structure
            dataset-root/
            └── annot_dir/
                ├── images/
                │   ├── image0001.jpg
                │   ├── image0002.jpg
                │   └── ...
                └── labels/
                    ├── image0001.txt
                    ├── image0002.txt
                    └── ...

        The dataset format is similar to YOLOv5 PyTorch Format which also resembles with PASCAL VOC Data Format

        Returns:
            List[Tuple[str]]: returns the image and label path combined for each sample
        """
        annotations = []
        annot_files = glob(self.annot_dir + "labels/*txt")

        errors = 0
        for annot_file in annot_files:
            image_file = annot_file.replace("txt", "jpg").replace("labels", "images")
            if os.path.exists(image_file):
                annotations.append((image_file, annot_file))
            else:
                errors += 1

        print(f"{errors} Errors found in {self.annot_dir=}")
        print(f"{len(annotations)} records loaded")

        return annotations

    def __len__(self) -> int:
        """Defined length of the dataset, dunder method

        Returns:
            int: Length of the dataset
        """
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets a single record for the dataset

        Args:
            idx (int): Index of the sample

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Image Tensor and BBox Tensor
        """

        image_path, label_path = self.annotations[idx]

        image = Image.open(image_path)
        image_tensor = self.tt(image)

        with open(label_path, "r") as labelf:
            lines = labelf.readlines()
            bboxes = []
            for line in lines:
                bbox = list(map(float, line.split()))
                bboxes.append(bbox)

        # Creating a placeholder for the bounding boxes per grid
        # Each row will hold [C1, C2, ..., Cn, BBox_Prob, BBox_Center_X, BBox_Center_Y, BBox_W, BBox_H]
        bboxes_tensor = torch.zeros(self.S * self.S, self.C + 5)

        for bbox in bboxes:
            # The bboxes are relative as per each image
            # For training purpose we have to make them
            # relative per grid, total grid count is S * S

            # Each bbox carries -> [CLASS_LABEL, CENTER_X, CENTER_Y, BBOX_W, BBOX_H]
            class_label, x, y, w, h = bbox
            class_label = int(class_label)
            assert class_label < self.C, f"More Classes found than given {self.C=}"

            # Finding respective grid cell of teh object
            grid_x, grid_y = int(x * self.S), int(y * self.S)

            # Assign that grid_x and grid_y has a object and it's of class `class_label`
            cell_loc = grid_x * self.S + grid_y
            bboxes_tensor[cell_loc, self.C] = 1
            bboxes_tensor[cell_loc, class_label] = 1

            # Defining relative bbox coord as per grid
            bbox_x = (x * self.S) - grid_x
            bbox_y = (y * self.S) - grid_y

            # Defining relative height and width
            bbox_h = h * self.S
            bbox_w = w * self.S

            bboxes_tensor[cell_loc, self.C + 1 : self.C + 5] = torch.tensor(
                [bbox_x, bbox_y, bbox_w, bbox_h]
            )

        return image_tensor, bboxes_tensor


# if __name__ == "__main__":
#     root_dir = "datasets/thermal_cheetah/test/"
#     num_classes = 2
#     num_grids = 7
#     dataset = Yolov1Dataset(root_dir, num_classes, num_grids)

#     image, bboxes = dataset[2]
#     print(f"{image.shape=}")
#     print(f"{bboxes.shape=}")
