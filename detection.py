from picamera2 import Picamera2, Preview
from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from util.imagenet_classes import classes
import time
import os

# i don't think I need this
# torch.backends.quantized.engine = 'qnnpack'
WEIGHTS = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT

class ObjectDetection:

    def __init__(self):
        # Init the picamera
        self.picam2 = Picamera2()

        # init the model
        self.model = ssdlite320_mobilenet_v3_large(weights=WEIGHTS).eval()

        # preprocess function
        self.preprocess = WEIGHTS.transforms()

    def configure_camera(self, debug=False):
        # create default configuration
        config = self.picam2.create_preview_configuration()
        
        # configure the camera
        self.picam2.configure(config)

        if (debug):
            # start preview
            self.picam2.start_preview(Preview.QTGL)

        # start the camera
        self.picam2.start()

    def run_detection(self, debug=False):
        if (debug):
            # frame counting
            last_logged = time.time()
            last_logged_frame_count = 0
            frame_count = 0
        
        with torch.no_grad():
            while True:
                # capture PIL image using picamera
                if (debug):
                    print('[INFO] capturing image...')
                raw_iamge = self.picam2.capture_image().convert(mode='RGB')

                # preprocess the image
                if (debug):
                    print('[INFO] preprocessing image...')
                input_image = self.preprocess(raw_iamge)

                # adjust to correct batch size = 1
                input_image = input_image.unsqueeze(0)

                # forward image through the model
                if (debug):
                    print('[INFO] forwarding through model...')
                logits = self.model(input_image)

                # print model output
                predicted_labels_num = logits[0]['labels'][:10] # top 10 predicted outputs
                predicted_scores = logits[0]['scores'][:10] # top 10 predicted scores
                predicted_labels = [self.weights.meta['categories'][i] for i in predicted_labels_num]

                # print top model confidences and frame rate
                if (debug):
                    os.system('clear')
                    for i in range(len(predicted_labels)):
                        print(f'{predicted_labels[i]}: {predicted_scores[i] * 100:.1f}')

                    # log frame / performance
                    frame_count += 1
                    now = time.time()
                    print(f"{last_logged_frame_count} fps")
                    if (now - last_logged) > 1:
                        last_logged = now
                        last_logged_frame_count = frame_count
                        frame_count = 0

                time.sleep(1)
                