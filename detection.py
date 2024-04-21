from picamera2 import Picamera2, Preview
from PIL import Image
import torch
import torchvision
from torchvision.transforms import v2
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
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

    
    def run_detection(self, detected, detectedLock, debug=False):
        if (debug):
            # frame counting
            last_logged = time.time()
            last_logged_frame_count = 0
            frame_count = 0
        
        with torch.no_grad():                
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
            predicted_labels = [WEIGHTS.meta['categories'][i] for i in predicted_labels_num]

            # determine if there is a person inside the predictions
            detectedLock.acquire() # acquire detected lock
            if (predicted_scores['person'] > 0.5):
                detected[0] = True
            else:
                detected[0] = False
            detectedLock.release() # release the lock

            # print top model confidences and frame rate
            if (debug):
                # clear terminal screen
                    os.system('clear')

                    # print detection status
                    detectedLock.acquire()
                    print(f'PERSON DETECTED: {detected[0]}')
                    detectedLock.release()

                    # print top 5 scores
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