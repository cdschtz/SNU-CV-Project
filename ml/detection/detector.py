import os
from pathlib import Path

import cv2
import torch
from torchvision.io.image import read_image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights, SSDLite320_MobileNet_V3_Large_Weights,
    fasterrcnn_resnet50_fpn_v2, ssdlite320_mobilenet_v3_large)
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes
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

        self.model.eval()

        if result_folder is None:
            self.result_folder = "./ml/detection/results"
            self.result_folder_images = self.result_folder + "/images"

        self.batch = None
        self.img_data_folder = None

    def _load_video_images(self, path):
        vidcap = cv2.VideoCapture(path)

        success, image = vidcap.read()
        count = 0

        folder = "/".join(path.split("/")[:-1])  # Video folder path
        os.system(f"rm -rf {folder}/images")  # Remove exisiting images
        Path(f"{folder}/images").mkdir(parents=True, exist_ok=True)
        self.img_data_folder = f"{folder}/images"

        # For future reference:
        # Get number of necessary digits: (given no. of files)
        # int(np.ceil(np.log10(1000 + 1)))

        while success:
            # save frame as JPEG file in data/images folder
            cv2.imwrite(f"{self.img_data_folder}/frame{count:06d}.jpg", image)

            success, image = vidcap.read()
            count += 1

    def _preprocess_images(self):
        """Preprocess images for model inference."""
        assert (self.img_data_folder != None)
        self.batch = torch.tensor([], dtype=torch.float32)

        files = Path(self.img_data_folder).glob("*.jpg")

        preprocess = self.weights.transforms()

        for file in sorted(files):
            img = read_image(str(file))
            img = preprocess(img)
            self.batch = torch.cat((self.batch, img.unsqueeze(0)), dim=0)

    def _save_results(self, img, idx):
        Path(self.result_folder_images).mkdir(parents=True, exist_ok=True)
        img.save(f"{self.result_folder_images}/frame{idx:06d}.jpg")

    def detect(self, video_file_path, save_file_name="output.mp4", batch_process_size=5):
        print(f"Detecting with batch size of: {batch_process_size}")
        self._load_video_images(video_file_path)
        self._preprocess_images()

        n = self.batch.shape[0]
        for i in range(0, n, batch_process_size):
            print("Iteration: ", i+1)

            batch = self.batch[i:i + batch_process_size]
            predictions = self.model(batch)

            for idx, prediction in enumerate(predictions):
                labels = [self.weights.meta["categories"][i]
                          for i in prediction["labels"]]
                img = read_image(
                    f"{self.img_data_folder}/frame{i+idx:06d}.jpg")
                box = draw_bounding_boxes(
                    img,
                    boxes=prediction["boxes"],
                    labels=labels,
                    colors="red",
                    width=4, font_size=30
                )

                im = to_pil_image(box.detach())
                self._save_results(im, i+idx)
