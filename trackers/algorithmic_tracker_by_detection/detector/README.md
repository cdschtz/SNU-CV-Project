
# Steps to replicate

Link to detection models (and size comparison): <https://pytorch.org/vision/stable/models.html#object-detection>

## Installation and setup

For this code to work you need to have [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) installed on your machine. After installing Conda, run the following commands from the root of this repository (i.e., not here):

```bash
conda create -n snu-cv-project python=3.9

conda activate snu-cv-project
```

```bash
pip install -r ./trackers/algorithmic_tracker_by_detection/requirements.txt
```

Now, to run the detector model that generates the detections images and saves them in the results folder, use the following command (with the activated conda environment):

```bash
python ./trackers/algorithmic_tracker_by_detection/main.py <PATH_TO_VIDEO_FILE>
```

where you replace <PATH_TO_VIDEO_FILE> by the path to the video file you want to run the detector on. An exemplary video can be found in the [data folder](../../data/) folder.

To run it on the test video you can use:

```bash
python ./trackers/algorithmic_tracker_by_detection/main.py ./data/test_short_video.mov
```

This command by default will use the small and fast dector model ssd, if you want to use the more precise but slower fasterRCNN model you can append the `--model faster_rcnn` flag to the command:

```bash
python ./trackers/algorithmic_tracker_by_detection/main.py <PATH_TO_VIDEO_FILE> --model faster_rcnn
```
