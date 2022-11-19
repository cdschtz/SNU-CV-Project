from detector import Detector

if __name__ == '__main__':
    detector = Detector()
    detector.detect(
        "./ml/detection/data/test_short_video.mov",
        save_file_name="test_short_video.mp4"
    )
