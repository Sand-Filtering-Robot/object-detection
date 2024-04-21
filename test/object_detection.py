from picamera2 import Picamera2, Preview
from PIL import Image
import torch
import torchvision
from torchvision import models 
from torchvision.transforms import v2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
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
weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT

# torch image preprocessing
preprocess = weights.transforms()

# mobile net
detection_model = ssdlite320_mobilenet_v3_large(weights=weights).eval()

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
        output = detection_model(model_input)

        #time.sleep(DELAY)

        # print model output
        predicted_labels_num = output['labels'][:10] # top 10 predicted outputs
        predicted_scores = output['scores'][:10] # top 10 predicted scores
        predicted_labels = [weights.meta['categories'][i] for i in predicted_labels_num]

        os.system('clear')
        for i in range(len(predicted_labels)):
            print(f'{predicted_labels[i]}: {predicted_scores[i]}')
        
        # log frame / performance
        frame_count += 1
        now = time.time()
        print(f"{last_logged_frame_count} fps")
        if (now - last_logged) > 1:
            last_logged = now
            last_logged_frame_count = frame_count
            frame_count = 0
        









