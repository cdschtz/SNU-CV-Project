import os
from pathlib import Path

import cv2
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights, SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn_v2, ssdlite320_mobilenet_v3_large)
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import draw_bounding_boxes, draw_keypoints
from tqdm import tqdm
from utils import MODEL_VARIANT


class Detector:
    def __init__(self, result_folder=None, model_variant: MODEL_VARIANT = MODEL_VARIANT.FASTER_RCNN):
        if model_variant is MODEL_VARIANT.FASTER_RCNN:
            self.weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(
                weights=self.weights, box_score_thresh=0.9)
        elif model_variant is MODEL_VARIANT.SSD_LITE:
            self.weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            self.model = ssdlite320_mobilenet_v3_large(
                weights=self.weights, score_thresh=0.5)

        assert self.weights is not None
        assert self.model is not None
        print(f"Detector model: {model_variant}")

        self.model.eval()

        if result_folder is None:
            self.result_folder = "./ml/detection/results"

        self.batch = None
        self.img_data_folder = None
        self.num_frames = None

    def _load_video_images(self, path):
        """Parses video file and saves all frames as images."""
        assert self.video_file_name is not None

        vidcap = cv2.VideoCapture(path)
        success, image = vidcap.read()
        count = 0

        folder = "/".join(path.split("/")[:-1])  # Video folder path
        self.img_data_folder = f"{folder}/{self.video_file_name}"

        os.system(f"rm -rf {self.img_data_folder}")  # Remove exisiting images
        Path(f"{self.img_data_folder}").mkdir(parents=True, exist_ok=True)

        # For future reference:
        # Get number of necessary digits: (given no. of files)
        # int(np.ceil(np.log10(1000 + 1)))

        while success:
            # save frame as JPEG file in data/images folder
            cv2.imwrite(f"{self.img_data_folder}/frame{count:06d}.jpg", image)

            success, image = vidcap.read()
            count += 1

        self.num_frames = count

    def _preprocess_images(self, files) -> torch.Tensor:
        """Preprocess images for model inference."""
        batch = torch.tensor([], dtype=torch.float32)

        preprocess = self.weights.transforms()
        for i, file in enumerate(sorted(files)):
            img = read_image(str(file))
            img = preprocess(img)
            batch = torch.cat((batch, img.unsqueeze(0)), dim=0)

        return batch

    def _save_image_results(self, img, idx):
        folder = Path(self.result_folder + "/" +
                      self.video_file_name + "/" + self.model.__class__.__name__)
        Path(folder).mkdir(parents=True, exist_ok=True)
        img.save(f"{folder}/frame{idx:06d}.jpg")

    def _serialize_detections(self, detections):
        """Serialize detections to .txt file."""
        folder = Path(self.result_folder + "/" +
                      self.video_file_name + "/" +
                      self.model.__class__.__name__ + "_detections")
        Path(folder).mkdir(parents=True, exist_ok=True)

        with open(f"{folder}/detections.txt", "w") as f:
            for detection in detections:
                f.write(",".join(map(lambda x: str(x), list(detection.values()))))
                f.write("\n")
            f.close()

    def detect(self, video_file_path, batch_process_size=5):
        print(f"Starting detection with batch size of: {batch_process_size}")
        self.video_file_name = video_file_path.split(
            "/")[-1].split(".")[0]  # without extension
        self._load_video_images(video_file_path)

        assert (self.img_data_folder != None)
        assert (self.num_frames != None)
        files = Path(self.img_data_folder).glob("*.jpg")
        n = self.num_frames

        all_detections = []

        batch_files = []
        for i, file in enumerate(tqdm(sorted(files))):
            batch_files.append(file)

            if (i + 1) % batch_process_size == 0 or i == n - 1:
                batch = self._preprocess_images(batch_files)
                batch_files = []

                predictions = self.model(batch)

                for idx, prediction in enumerate(predictions):
                    dist = min(4, len(batch) - 1)

                    labels = [self.weights.meta["categories"][i]
                              for i in prediction["labels"]]
                    img = read_image(
                        f"{self.img_data_folder}/frame{i-dist+idx:06d}.jpg")

                    for score, box in zip(prediction["scores"], prediction["boxes"]):
                        detection = {}
                        detection["frameNumber"] = i-dist+idx
                        detection["objectId"] = -1

                        # top-left corner
                        detection["x0"] = round(box[0].item(), 2)
                        detection["y0"] = round(box[1].item(), 2)

                        # bottom-right corner
                        detection["x1"] = round(box[2].item(), 2)
                        detection["y1"] = round(box[3].item(), 2)

                        detection["confidenceScore"] = round(score.item(), 2)
                        all_detections.append(detection)

                    box = draw_bounding_boxes(
                        img,
                        boxes=prediction["boxes"],
                        labels=labels,
                        colors="red",
                        width=4, font_size=30
                    )

                    im = to_pil_image(box.detach())

                    self._save_image_results(im, i-dist+idx)

        self._serialize_detections(all_detections)
