import unittest
import torch

from yolo.models.detector import YOLOv1


class YOLOv1Test(unittest.TestCase):

    def test_output_shape(self):
        C = 20
        S = 7
        B = 2
        IN_CHANNELS = 3
        BATCH_SIZE = 3
        IMG_SHAPE = (448, 448)
        yolo_obj = YOLOv1(IN_CHANNELS, C)
        rand_data = torch.randn(BATCH_SIZE, IN_CHANNELS, *IMG_SHAPE)
        assert yolo_obj(rand_data).shape == torch.Size([BATCH_SIZE, S**2, (B * 5 + C)])


if __name__ == "__main__":
    unittest.main()
