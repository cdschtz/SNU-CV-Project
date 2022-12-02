import argparse

from detector import Detector
from utils import MODEL_VARIANT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Path to the video file', type=str)
    parser.add_argument('--model', type=str,
                        default=MODEL_VARIANT.SSD_LITE, help='Model name')
    args = parser.parse_args()

    detector = Detector(model_variant=MODEL_VARIANT(args.model))
    detector.detect(args.file)
