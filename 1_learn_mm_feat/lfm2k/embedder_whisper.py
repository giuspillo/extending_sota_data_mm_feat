import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
import whisper
import numpy as np
import ffmpeg

def load_audio(file_path, sr=16000):
    """
    Load audio from a video file using ffmpeg
    """
    try:
        # Use ffmpeg to extract audio from video file
        out, _ = (
            ffmpeg.input(file_path)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"Error extracting audio from {file_path}: {e.stderr.decode()}")
        return None

    # Convert to numpy array
    audio = np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
    return audio

def get_whisper_embedding(model, audio_array, device="cuda:0", chunk_duration=30):
    """
    Extract Whisper embeddings from the entire audio file
    """
    sample_rate = 16000
    chunk_length = chunk_duration * sample_rate  # Length of each chunk in samples
    
    embeddings = []

    # Process audio in chunks
    for start in range(0, len(audio_array), chunk_length):
        end = min(start + chunk_length, len(audio_array))
        audio_chunk = audio_array[start:end]
        
        # If the chunk is too short, pad it
        if len(audio_chunk) < chunk_length:
            audio_chunk = np.pad(audio_chunk, (0, chunk_length - len(audio_chunk)))

        # Convert to mel spectrogram
        mel = whisper.log_mel_spectrogram(audio_chunk).to(device)
        
        # Extract embeddings using Whisper model
        with torch.no_grad():
            mel = mel.unsqueeze(0)  # Add batch dimension
            embeddings_chunk = model.encoder(mel)
            
            # Average pooling over time dimension to get a single embedding for this chunk
            chunk_embedding = torch.mean(embeddings_chunk, dim=1).squeeze(0)
            embeddings.append(chunk_embedding.cpu().numpy())

    # Concatenate embeddings from all chunks to get the final embedding
    final_embedding = np.mean(embeddings, axis=0)
    
    return final_embedding

# Load Whisper model
def load_whisper_model(model_size):
    model = whisper.load_model(model_size)
    return model

# Select model size based on your needs
model_size = "base"  # Options: "tiny", "base", "small", "medium", "large"
whisper_model = load_whisper_model(model_size).to("cuda:0")

files_dir = "_songs/"
audios = [os.path.join(files_dir, x) for x in sorted(os.listdir(files_dir))]
audio_embeddings = {}

# Process each audio file
for audio_path in tqdm(audios):
    name = os.path.basename(audio_path).split('.')[0]
    
    audio_array = load_audio(audio_path)
    
    if audio_array is None or len(audio_array) == 0:
        print(f"Skipping {name} - could not extract audio")
        continue
    
    # Get Whisper embedding for the entire audio
    embedding = get_whisper_embedding(whisper_model, audio_array)
    embedding = np.array(embedding)

    # Store embedding
    audio_embeddings[name] = embedding
    # print(embedding.shape)
    # print(name, type(name))

    # break

# Save the embeddings to a pickle file
pickle.dump(audio_embeddings, open('songs/whisper.pkl', 'wb'))
