from detector import Detector
from utils import MODEL_VARIANT

if __name__ == '__main__':
    detector = Detector(model_variant=MODEL_VARIANT.SSD_LITE)
    detector.detect(
        "./ml/detection/data/test_short_video.mov",
        save_file_name="test_short_video.mp4"
    )
