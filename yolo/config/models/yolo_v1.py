# Configuratiosn for YOLOv1

NUM_ANCHORS = 2
NUM_SEGMENTS = 7

DARKNET = [
    # fmt: off
    ["C", 64, 7, 2, 1],  # Conv2d Layer (type, out_channels, kernel_size, stride, padding)
    # fmt : on
    ["M", 2, 2],  # MaxPool2d Layer (type, kernel_size, stride)
    ["C", 192, 3, 1, 1],
    ["M", 2, 2],
    ["C", 128, 1, 1, 0],
    ["C", 256, 3, 1, 1],
    ["C", 256, 1, 1, 0],
    ["C", 512, 3, 1, 1],
    ["M", 2, 2],
    ["R", 4, (["C", 256, 1, 1, 0], ["C", 512, 3, 1, 1])],
    ["C", 512, 1, 1, 0],
    ["C", 1024, 3, 1, 1],
    ["M", 2, 2],
    ["R", 2, (["C", 512, 1, 1, 0], ["C", 1024, 3, 1, 1])],
    ["C", 1024, 3, 1, 1],
    ["C", 1024, 3, 2, 1],
    ["C", 1024, 3, 1, 1],
    ["C", 1024, 3, 1, 1],
]
