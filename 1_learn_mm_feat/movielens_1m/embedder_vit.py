from transformers import AutoImageProcessor, ViTModel
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
import os
import pickle


# configuration = ViTConfig()

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
print(model.config)

embs_cls = dict()
embs_avg = dict()
paths = os.listdir('_posters/')
print('Number of images:', len(paths))

for path in tqdm(paths, total=len(paths)):

  try:

    # name = int(path.split('_')[1].split('.')[0])
    name = int(path.split('.')[0].split('_')[1])

    image = Image.open('images/images/'+path).convert('RGB')
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
      outputs = model(**inputs)

    last_hidden_states = (outputs.last_hidden_state)

    cls = last_hidden_states[:, 0, :].reshape(-1)
    avg = last_hidden_states[:, 1:, :].mean(dim=1).reshape(-1)

    embs_cls[name] = np.array(cls)
    embs_avg[name] = np.array(avg)

    # print(path, '\n', name, '\n', type(name))
    # print(cls.shape)
    # print(avg.shape)

    # break
  
  except Exception as e:
    print(f'failed processing with {path}')

pickle.dump(embs_cls, open('posters/vit_cls.pkl', 'wb'))
pickle.dump(embs_avg, open('posters/vit_avg.pkl', 'wb'))