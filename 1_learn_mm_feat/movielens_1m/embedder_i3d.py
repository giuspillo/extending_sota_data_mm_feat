import os
import torch
import random
import torchvision.transforms
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.io.video_reader import VideoReader
from tqdm import tqdm
from itertools import islice, takewhile
import pickle

def seed_everything(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    return seed

# Load I3D model
# I3D-ResNet50 from PyTorchVideo
def load_i3d_model():
    # Import here to keep dependencies modular
    from pytorchvideo.models.hub import i3d_r50
    
    # Load pretrained I3D model (based on ResNet-50 backbone)
    model = i3d_r50(pretrained=True)
    
    # Define feature extraction layer (typically before the final classification layer)
    # For ResNet-based models, we typically want features before the final FC layer
    feature_layer = "blocks.5"  # This may need adjustment based on the specific model implementation
    
    # Create feature extractor
    feature_extractor = create_feature_extractor(model, {feature_layer: "feature_layer"})
    feature_extractor = feature_extractor.to('cuda:0').eval()
    
    # Freeze parameters
    for params in feature_extractor.parameters():
        params.requires_grad = False
        
    return feature_extractor

# Load the I3D model
model = load_i3d_model()

# Define preprocessing specific to I3D
def preprocess_frames(frames):
    # Convert frames to float and scale to [0, 1]
    frames = frames.to(torch.float32) / 255.0
    
    # Resize to appropriate dimensions for I3D (typically 224x224)
    frames = torchvision.transforms.Resize((224, 224))(frames)
    
    # Normalize using RGB ImageNet mean and std
    frames = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(frames)
    
    return frames

files_dir = "_videos/"
videos = [os.path.join(files_dir, x) for x in sorted(os.listdir(files_dir))]
videos_outs = {}

for video_path in tqdm(videos):
    name = int(os.path.basename(video_path).split('.')[0])
    # print(name)
    
    # reseed everything so that the order of the videos doesn't matter
    seed_everything(42)
    
    reader = VideoReader(video_path)
    frame_iter = (frame['data'] for frame in takewhile(lambda x: x['pts'] <= 30, reader.seek(0)))
    
    clips_batch = []
    for _ in range(5):
        # I3D typically uses 64 frames, but we can adjust this
        clip_frames = list(islice(frame_iter, 64))
        if len(clip_frames) < 64:  # Skip if we don't have enough frames
            # Optionally pad with zeros to reach 64 frames
            if len(clip_frames) > 32:  # Only pad if we have a reasonable number of frames
                padding = [torch.zeros_like(clip_frames[0]) for _ in range(64 - len(clip_frames))]
                clip_frames.extend(padding)
            else:
                continue
        
        clip = torch.stack(clip_frames)
        clip = preprocess_frames(clip)
        clips_batch.append(clip)
    
    if len(clips_batch) == 0:
        continue
    
    # Stack the clips
    clips_batch = torch.stack(clips_batch)
    
    # I3D expects input shape [batch_size, channels, num_frames, height, width]
    # Current shape is [batch_size, num_frames, channels, height, width]
    clips_batch = clips_batch.permute(0, 2, 1, 3, 4)
    
    with torch.no_grad():
        # Forward pass through I3D model
        embeddings = model(clips_batch.to('cuda:0'))['feature_layer']
        
        # Global average pooling over spatial and temporal dimensions
        # Output shape will be [batch_size, channels]
        embeddings = torch.mean(embeddings, dim=[2, 3, 4])
        
        # Apply mean pooling across clips to get a single video embedding
        video_embedding = torch.mean(embeddings, dim=0)
    
    # Store the single embedding vector for this video
    video_embedding = np.array(video_embedding.cpu())
    videos_outs[int(name)] = video_embedding

    # break
    

# Save the embeddings
print(len(videos_outs), videos_outs[next(iter(videos_outs))].shape)
pickle.dump(videos_outs, open('videos/i3d.pickle', 'wb'))

# run with CUBLAS_WORKSPACE_CONFIG=:4096:8 python embedder_i3d.py 