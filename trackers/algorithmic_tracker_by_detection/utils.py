from enum import Enum


class MODEL_VARIANT(Enum):
    FASTER_RCNN = "faster_rcnn"  # num_params: 43,712,278 -> larger, slower, more accurate
    SSD_LITE = "ssd_lite"  # num_params: 3,440,060 -> smaller, faster, less accurate
