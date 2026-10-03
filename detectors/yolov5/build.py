"""
YOLOv5 builders for the three tasks.

Architecture (algorithm-specific, this folder):
    CSPDarknet (C3 blocks, SiLU, SPPF) + PaFPN neck, three scales (8, 16, 32)
    -> yolov5_backbone.py / yolov5_pafpn.py / yolov5_basic.py / yolov5_neck.py
       copied unchanged from the Legacy Image_Yolo_Yolov5 project
    -> head: one 1x1 convolution per scale (original YOLOv5 "Detect")
    -> size: VARIANT (legacy projects used "yolov5_n")

Detection layers: reused unchanged from detectors/yolov3, exactly as v2 / v4.
Anchors: taken from the v3 cfg files (Image: COCO anchors, KITTI: v3 anchors).

Not used from the Legacy v5 projects: the decoupled head, the YOLOv5
anchor-ratio matcher and its loss (replaced by the shared YOLO layers), so v5
differs from v3 in backbone / neck only.
"""


VARIANT = "yolov5_n"
