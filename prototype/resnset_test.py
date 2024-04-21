import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as t
import json
from glob import glob
import os
from PIL import Image, ImageFile
from imagenet_classes import classes

ImageFile.LOAD_TRUNCATED_IMAGES = True # required for Pillow
image_path = 'images/momo_2.jpg' # maybe automate this to run on multiple images using glob

mean = (0.485 + 0.456 + 0.406) / 3
std = (0.229 + 0.224 + 0.225) / 3

print(f'opening image from path {image_path}')
image = Image.open(image_path)
resized_im = image.resize((256, 256))
#image.show()
#resized_im.show()

weights = torchvision.models.ResNet50_Weights.DEFAULT
net = torchvision.models.resnet50(weights=weights)
preprocess = weights.transforms()

custom_preprocess = t.Compose([
    t.PILToTensor(),
    t.CenterCrop(size=[224]),
    t.Resize(size=[256], antialias=True),
    t.ToDtype(torch.float32),
    t.Normalize(mean=[mean], std=[std]),
])
print(preprocess)
#print(custom_preprocess)

print(f'preprocessing image...')
processed_image = preprocess(image)
net_input = processed_image.unsqueeze(0)
print(processed_image)
#net_input = net_input.repeat(1, 3, 1, 1)
print(net_input.size())

torch_image = F.pil_to_tensor(image)
#plt.imshow(resized_im)
#plt.show()

#plt.imshow(processed_image.permute(1, 2, 0))
#plt.show()

print(f'running model...')
net = net.eval()
raw_output = net(net_input)

log_softmax = nn.Softmax(dim=1)
output = log_softmax(raw_output)
print(output[0])


top = list(enumerate(output[0]))

total_sum = 0
for idx, val in top:
    total_sum += val.item()
print(total_sum)

top.sort(key=lambda x: x[1], reverse=True)
for idx, val in top[:10]:
    print(f"{val.item()*100:.2f}% {classes[idx]}")