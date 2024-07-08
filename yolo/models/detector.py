"""Detector Module"""

import torch
import torch.nn as nn
from typing import List, Any

from yolo.config.models import yolo_v1
from yolo.models import layers


class YOLOv1(nn.Module):

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.in_channels: int = in_channels
        self.C: int = num_classes
        self.B: int = yolo_v1.NUM_ANCHORS
        self.S: int = yolo_v1.NUM_SEGMENTS
        self.out_shape: int = self.C + self.B * 5
        self.darknet: nn.Module = self._create_darknet(yolo_v1.DARKNET)
        self.fcs: nn.Module = self._create_fcs()
        self.backbone: nn.Sequential = nn.Sequential()
        self.create_backbone()

    def _create_darknet(self, confguration: List[List[Any]]) -> nn.Module:
        """Generated the darknet-24 layer

        Args:
            confguration (List[List[Any]]): Configuration for the Darknet

        Returns:
            nn.Module: Darknet model
        """
        network = nn.ModuleList()

        in_channels: int = self.in_channels
        for layer in confguration:
            if layer[0] == "C":
                network.append(layers.Conv(in_channels, *layer[1:]))
                in_channels = layer[1]
            elif layer[0] == "M":
                network.append(nn.MaxPool2d(*layer[1:]))
            elif layer[0] == "R":
                for _ in range(layer[1]):
                    for sub_layer in layer[2]:
                        if sub_layer[0] == "C":
                            network.append(layers.Conv(in_channels, *sub_layer[1:]))
                            in_channels = sub_layer[1]
                        elif layer[0] == "M":
                            network.append(nn.MaxPool2d(*sub_layer[1:]))

        return nn.Sequential(*network)

    def _create_fcs(self) -> nn.Sequential:
        """Creates the Head of the Backbone using fully-connected layers

        Returns:
            nn.Sequential: Fully-Connected head
        """
        # Hidden dim is the only configurable command
        # So we have kept that abstracted
        fcs = nn.Sequential(
            nn.Linear(
                1024 * self.S * self.S, 256
            ),  # Actual is 1024 , for computational ease, we're giving 256
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            nn.Linear(256, self.out_shape * self.S**2),
        )

        return fcs

    def create_backbone(self) -> nn.Sequential:
        """Creates the backbone for YOLOv1

        Returns:
            nn.Sequential: BackBone model created by joining the conv and fc layers
        """
        self.backbone.extend(self.darknet)
        self.backbone.add_module("flatten", nn.Flatten())
        self.backbone.extend(self.fcs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        flattened_tensor = self.backbone(inputs)
        reshaped_tensor = flattened_tensor.reshape(-1, self.S**2, self.out_shape)
        return reshaped_tensor


# if __name__ == "__main__":

#     rand_data = torch.randn(2, 3, 448, 448)
#     model = YOLOv1(3, 20)
#     output = model(rand_data)
#     # print(model)
#     print(output.shape)
