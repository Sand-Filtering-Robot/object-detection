from picamera2 import Picamera2, Preview
from PIL import Image
import torch
import torchvision
from torchvision import models 
from torchvision.transforms import v2
from imagenet_classes import classes
from env import verbose
import time
import os

# setup picamera
picam2 = Picamera2()
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)

# start the preview
picam2.start_preview(Preview.QTGL)

# start the camera
picam2.start()

# torch setup
torch.backends.quantized.engine = 'qnnpack'
weights = models.MobileNet_V3_Small_Weights.DEFAULT

# image preprocessing
preprocess = weights.transforms()

# mobile net
mobile_net = models.mobilenet_v3_small(weights=weights).eval()

# frame counting
last_logged = time.time()
last_logged_frame_count = 0
frame_count = 0

with torch.no_grad():
    while True:
        # capture an image
        if verbose:
            print('[INFO] capturing image...')
        raw_image = picam2.capture_image().convert(mode='RGB')

        # pre process the image
        if verbose:
            print('[INFO] preprocessing image...')
        model_input = preprocess(raw_image)

        # adjust to mini-batch size of 1
        model_input = model_input.unsqueeze(0)

        # run model :)
        if verbose:
            print('[INFO] running model...')
        output = mobile_net(model_input)

        #time.sleep(DELAY)

        # print model output
        top = list(enumerate(output[0].softmax(dim=0)))
        top.sort(key=lambda x: x[1], reverse=True)
        os.system('clear')
        for idx, val in top[:10]:
            print(f'{val.item()*100:.2f}% {classes[idx]}')


        # log frame / performance
        frame_count += 1
        now = time.time()
        print(f"{last_logged_frame_count} fps")
        if (now - last_logged) > 1:
            last_logged = now
            last_logged_frame_count = frame_count
            frame_count = 0
        









