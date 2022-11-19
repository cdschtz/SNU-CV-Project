# Steps to replicate

Link to detection models (and size comparison): <https://pytorch.org/vision/stable/models.html#object-detection>

### Create and activate a conda environment

```
conda create -n snu-cv-project python=3.9

conda activate snu-cv-project
```

### Install Python requirements

Install pytorch:

```
pip install torch torchvision torchaudio
```

Install Yolo (Object Detection Model) dependencies (**one line!**):

```
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
```

You might need to install additional dependencies in order to run Jupyter notebooks.

### Run the demo .ipynb file
