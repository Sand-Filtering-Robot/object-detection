# Object Detection

This repository holds all of the source code needed to run SandE's object detection functionality used to avoid obstacles.

## Dependencies

Hardware:
1. Raspberry Pi (preferably model 4B)
2. Pi Camera V3

Software:
- Picamera2: follow the oficial [raspberry pi camera setup documentation](https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf). 
- Pytorch: required in order to instantiate and use the pre-trained model. Run the following to install:
  ``` pip install torch torchvision ```

  Note: It is recommended to use a package manager such as Conda or venv for installing the required dependencies.

## Bugs & Issues
There were some noticeable issues during the setup and testing stages, mainly with certain repository dependencies and issues with python venv. Some of these issues are listed down belown alongside a possible quick fix:

- python venv initialization: make sure to enable the `--system-site-packages` flag during the creation of a python venv
