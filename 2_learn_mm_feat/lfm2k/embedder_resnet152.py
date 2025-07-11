import torch
import torchvision.transforms as transforms
from torchvision.models import resnet152
from PIL import Image
import os
from tqdm import tqdm
import pickle
import numpy as np

# Load pre-trained ResNet-152 model
resnet_model = resnet152(pretrained=True)
resnet_model.eval()  # Set the model to evaluation mode

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# for all the images
paths = os.listdir('_covers/')
print('Number of images:', len(paths))

resnet_embs = dict()

for path in tqdm(paths, total=len(paths)):

  # name = int(path.split('_')[1].split('.')[0])
  name = path.split('.')[0]

  try:
  
    image = Image.open('_covers/'+path).convert('RGB')
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Get the image embedding
    with torch.no_grad():
        output = resnet_model(input_image)

    # Extract the embedding from the output
    embedding = np.array(output.squeeze())

    resnet_embs[name] = embedding
  except Exception as e:
     print(f'failed processing with {path}')

  # print(embedding.shape)
  # print(path, name, type(name))
  # break


pickle.dump(resnet_embs, open('covers/resnet152.pkl', 'wb'))

