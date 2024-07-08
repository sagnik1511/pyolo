import torch


def iou_score(pred_bbox: torch.Tensor, taregt_bbox: torch.Tensor) -> torch.Tensor:
    """Generates ios score in batches
        IOU(A, B) = Intersection(A, B) / Union(A, B)

        The input data comes as [CX_1, CY_1, W_1, H_1], [CX_2, CY_2, W_2, H_2]
        Where CX -> center point x-axis value of the object
              CY -> center point y-axis value of the object
              W -> width of the object
              H -> Height of the object

    Args:
        pred_bbox (torch.Tensor): First bounding box
        taregt_bbox (torch.Tensor): Second bounding box

    Returns:
        torch.Tensor: IOU Score in Batches
    """

    # Extracting the values from bbox_coords
    CX1 = pred_bbox[..., 0:1]
    CY1 = pred_bbox[..., 1:2]
    W1 = pred_bbox[..., 2:3]
    H1 = pred_bbox[..., 3:4]

    CX2 = taregt_bbox[..., 0:1]
    CY2 = taregt_bbox[..., 1:2]
    W2 = taregt_bbox[..., 2:3]
    H2 = taregt_bbox[..., 3:4]

    # To solve this we calculate the corner points of the BBox
    # We define them as T (Top) and D (Down)
    #   (XT, YT)
    #       +-----------+
    #       |           |
    #       |           |
    #       |           |
    #       |           |
    #       |           |
    #       +-----------+
    #                 (XD, YD)

    XT1 = CX1 - (W1 / 2)
    YT1 = CY1 - (H1 / 2)

    XD1 = CX1 + (W1 / 2)
    YD1 = CY1 + (H1 / 2)

    # Same for second bounding box
    XT2 = CX2 - (W2 / 2)
    YT2 = CY2 - (H2 / 2)

    XD2 = CX2 + (W2 / 2)
    YD2 = CY2 + (H2 / 2)

    # Finding the common area coordinates
    x1 = torch.max(XT1, XT2)
    y1 = torch.max(YT1, YT2)

    x2 = torch.min(XD1, XD2)
    y2 = torch.min(YD1, YD2)

    # caculating the area
    # Edge Case: If the BBox don't intersect, the area should be 0
    # We are using clamp function to mimic that scenario
    intersection_bbox_area = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)

    # Total Area of the BBoxes both included
    sum_bbox_area = (W1 * H1) + (W2 * H2)

    # Calculating Union(A, B)
    # A ∪ B = (A + B) - (A ∩ B)
    union_bbox_area = sum_bbox_area - intersection_bbox_area

    return intersection_bbox_area / (
        union_bbox_area + 1e-9
    )  # Adding the small value for numerical stability
