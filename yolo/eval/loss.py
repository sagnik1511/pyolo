import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Any

from yolo.eval.metrics import iou_score


class YOLOv1Loss(nn.Module):

    def __init__(
        self,
        num_classes: int,
        num_anchors: int = 2,
    ):
        super().__init__()
        """
        Args:
            num_classes (int): Number of Classes
            num_anchors (int): Number of Anchors/BBoxes per grid
        """
        self.B = num_anchors
        self.C = num_classes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def _fetch_best_bbox(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Fetches the best bboxes per cel per image in the prediction batch

        Args:
            preds (torch.Tensor): prediction tensor having a single pred per grid
            targets (torch.Tensor): true labels

        Returns:
            torch.Tensor: _description_
        """

        # Find the best box using IOU Score
        # To find the best box we have to calculate fetch the bbox
        # with the best iou score. YoloV1 expects 2 anchors
        # but we will setup the code for n anchors where n >= 1

        ious = torch.tensor([])
        target_bboxes = targets[..., self.C + 1 : self.C + 5]
        for anchor_idx in range(self.B):
            Start = self.C + anchor_idx * 5 + 1
            End = self.C + anchor_idx * 5 + 5
            print(f"Taking bbox -> [..., {Start} : {End}]")
            bbox_coords = preds[
                ..., self.C + anchor_idx * 5 + 1 : self.C + anchor_idx * 5 + 5
            ]
            anchor_iou = iou_score(bbox_coords, target_bboxes)
            ious = torch.cat([ious, anchor_iou], dim=2)

        _, best_bbox = torch.max(ious, dim=2)

        # Creating indexing method to extract the best box from preds
        best_bbox = best_bbox.squeeze(-1)
        best_bbox_start_indices = self.C + best_bbox * 5

        # Defining bbox indices 0 - 4 to extract each bbox
        filtered_indices = (
            torch.arange(5).view(1, 1, 5).expand(preds.shape[0], preds.shape[1], 5)
        )
        # Defining abolute bbox position
        best_bbox_indices = best_bbox_start_indices.unsqueeze(-1) + filtered_indices

        # Defining batch level indices for slicing
        batch_indices = (
            torch.arange(preds.shape[0])
            .view(-1, 1, 1)
            .expand(preds.shape[0], preds.shape[1], 5)
        )
        # Defining cell / grid level indices for slicing
        cell_indices = (
            torch.arange(preds.shape[1])
            .view(1, -1, 1)
            .expand(preds.shape[0], preds.shape[1], 5)
        )

        # Slicing the tensor on te expected values to retrieve a single bbox per cell
        pred_bboxes = preds[batch_indices, cell_indices, best_bbox_indices]

        return pred_bboxes

    def _calculate_bbox_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the BBox Loss

        BBox Loss per cell : lambda_coord * I * [(X_1 - X_2)**2 + (Y_1 - Y_2)**2] +
                            lambda_coord * I * [(sqrt(W_1) - sqrt(W_2))**2 + (sqrt(H_1) - sqrt(H_2))**2]

        Args:
            preds (torch.Tensor): prediction tensor having a single pred per grid
            targets (torch.Tensor): true bbox labels

            Both are having the following format -> [BATCH, S * S, CONF, X, Y, W, H]

        Returns:
            torch.Tensor: Bounding Box Loss per grid / cell
        """

        # defining if object is present or not
        obj_exist = targets[..., 0]

        # Defining sqrt of height and width with added numerical stabilities
        preds[..., 2:4] = (
            obj_exist.unsqueeze(-1)
            * torch.sign(preds[..., 2:4])
            * torch.sqrt(torch.abs(preds[..., 2:4]) + 1e-9)
        )
        targets[..., 2:4] = obj_exist.unsqueeze(-1) * torch.sqrt(
            targets[..., 2:4] + 1e-9
        )

        # Calculating box loss
        # Reshaping the tensors (BS, S*S, 4) -> (BS * S * S, 4)
        bbox_loss = F.mse_loss(
            torch.flatten(preds[..., 1:5], end_dim=-2),
            torch.flatten(targets[..., 1:5], end_dim=-2),
        )

        return bbox_loss

    def _calculate_class_object_loss(
        self, preds: torch.Tensor, best_bboxes: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the obkect loss per grid cell

        Args:
            preds (torch.Tensor): prediction tensor
            best_bboxes (torch.Tensor): prediction tensor having a single pred per grid i.e. the best boxes hyperparams
            targets (torch.Tensor): true label tensor

        Returns:
            torch.Tensor: Objectness Loss per grid
        """
        # Defining if object is present or not
        obj_exist = targets[..., 0]

        # Defining loss parameters
        obj_class_loss = F.mse_loss(
            torch.flatten(obj_exist * best_bboxes[..., 0]),
            torch.flatten(obj_exist * targets[..., self.C]),
        )

        # As we want the class confidence we're extracting that for further calculations
        # The class confdence is the first value (at 0th-ndex) of the 5 bbox values
        class_conf_tensor = preds[..., self.C :][..., ::5]
        broadcasted_targets = targets[..., self.C : self.C + 1].expand_as(
            class_conf_tensor
        )

        noobj_class_loss = self.lambda_noobj * F.mse_loss(
            torch.flatten(
                (1 - obj_exist).unsqueeze(-1) * class_conf_tensor, end_dim=-2
            ),
            torch.flatten(
                (1 - obj_exist).unsqueeze(-1) * broadcasted_targets, end_dim=-2
            ),
        )
        # for anchor_idx in range(self.B):
        #     start_idx = self.C + (anchor_idx * 5)
        #     noobj_class_loss = noobj_class_loss + (
        #         self.lambda_noobj
        #         * F.mse_loss(
        #             torch.flatten((1 - obj_exist) * preds[..., start_idx]),
        #             torch.flatten((1 - obj_exist) * targets[..., self.C]),
        #         )
        #     )

        return obj_class_loss + noobj_class_loss

    def _calculate_class_confidence_loss(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Calculates the class confidence loss per grid

        Args:
            preds (torch.Tensor): the best boxes hyperparams
            targets (torch.Tensor): true label tensor

        Returns:
            torch.Tensor: Confidence Loss
        """

        # Defining if object is present or not
        obj_exist = targets[..., self.C].unsqueeze(-1)

        confidence_loss = F.mse_loss(
            torch.flatten(obj_exist * preds[..., : self.C], end_dim=-2),
            torch.flatten(obj_exist * targets[..., : self.C], end_dim=-2),
        )

        return confidence_loss

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculates the Total loss for YOLOV1 detector

        preds shape : [BATCH_SIZE, S * S,  (C + B * 5)]
        targets shape : [BATCH_SIZE, S * S, C + 5]

        Args:
            preds (torch.Tensor): prediction tensor
            targets (torch.Tensor): true label tensor

        Returns:
            torch.Tensor: Total Loss combined
        """
        best_bboxes = self._fetch_best_bbox(preds, targets)

        loss = self._calculate_bbox_loss(best_bboxes, targets[..., self.C :])

        loss += self._calculate_class_object_loss(preds, best_bboxes, targets)

        loss += self._calculate_class_confidence_loss(preds, targets)

        return loss


# if __name__ == "__main__":
#     pred_box = torch.tensor([[[0.3, 0.4, 0.5, 0.7]]])
#     target_box = torch.tensor([[[0, 0, 0.1, 0.1]]])

#     iou_score_res = iou_score(pred_box, target_box)

#     print(iou_score_res.shape)
#     print(iou_score_res)

#     batch_size = 2
#     S = 3
#     B = 4
#     C = 2
#     predbboxes = torch.rand(batch_size, S * S, C + B * 5)
#     targetbboxes = torch.rand(batch_size, S * S, C + 5)
#     loss_fn = YOLOv1Loss(C, B)
#     loss = loss_fn(predbboxes, targetbboxes)
#     print(loss.item())
