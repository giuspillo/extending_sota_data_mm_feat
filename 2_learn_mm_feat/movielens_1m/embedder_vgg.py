import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
import os
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg19(pretrained=True).to(device).eval()
model.classifier = torch.nn.Sequential(*list(model.classifier.children())[:5])  # Up to fc2
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# for all the images
paths = os.listdir('_posters/')
print('Number of images:', len(paths))

embs = {}

for path in tqdm(paths, total=len(paths)):

  # get the name (aka dbid) given by the string 'graphid_dbpediaid'
  # name = int(path.split('_')[1].split('.')[0])
  name = int(path.split('.')[0].split('_')[1])

  try:

    image = Image.open('images/images/'+path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
      emb = model(image)
    emb = emb.reshape(-1).cpu().numpy()
    embs[name] = emb
    # print(emb.shape)
    # print(path, name)
    
  except Exception as e:
     print(f'failed processing with {path}')

  # break

pickle.dump(embs, open('posters/vgg.pkl', 'wb'))


