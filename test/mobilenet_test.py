import torch
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as t
import json
from glob import glob
import os
from PIL import Image, ImageFile
from imagenet_classes import classes

ImageFile.LOAD_TRUNCATED_IMAGES = True # required for Pillow
image_path = 'images/stinky_1.jpg' # maybe automate this to run on multiple images using glob

print(f'opening image from path {image_path}')
image = Image.open(image_path)
resized_im = image.resize((round(image.size[0]*0.0625), round(image.size[1]*0.0625)))
#image.show()

weights = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
net = torchvision.models.mobilenet_v3_small(weights = weights).eval()

preprocess = weights.transforms()
#print(preprocess)

print(f'preprocessing image...')
processed_image = preprocess(image)
net_input = processed_image.unsqueeze(0)
#print(processed_image)

print(f'running model...')
output = net(net_input)

#print(output[0].softmax(dim=0))


top = list(enumerate(output[0].softmax(dim=0)))
top.sort(key=lambda x: x[1], reverse=True)
for idx, val in top[:10]:
    print(f"{val.item()*100:.2f}% {classes[idx]}")

torchvision.disable_beta_transforms_warning()