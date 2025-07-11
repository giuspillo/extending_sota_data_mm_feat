import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
import whisper
import numpy as np
import ffmpeg
from pydub import AudioSegment

# def mp4_to_wav(mp4_path):
#     """Converts an MP4 file to WAV format using a temporary file."""
#     audio = AudioSegment.from_file(mp4_path, format="mp4")
#     return audio

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
#         audio.export(temp_wav.name, format="wav")
#         return temp_wav.name  # Return temp WAV path

def extract_audio_from_mp4(mp4_path, sr=16000):
    """
    Extracts raw audio from an MP4 file using ffmpeg and returns it as a numpy array.
    """
    try:
        out, _ = (
            ffmpeg.input(mp4_path)
            .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        print(f"Error extracting audio from {mp4_path}: {e.stderr.decode()}")
        return None

    # Convert to float32 numpy array (normalized)
    audio = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
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

files_dir = "_videos/"
videos = [os.path.join(files_dir, x) for x in sorted(os.listdir(files_dir))]
audio_embeddings = {}

# Process each video file
for video_path in tqdm(videos):
    name = os.path.basename(video_path).split('.')[0]
    
    audio_array = extract_audio_from_mp4(video_path)
    
    if audio_array is None or len(audio_array) == 0:
        print(f"Skipping {name} - could not extract audio")
        continue
    
    # Get Whisper embedding for the entire audio
    embedding = get_whisper_embedding(whisper_model, audio_array)
    embedding = np.array(embedding)

    # Store embedding
    audio_embeddings[int(name)] = embedding
    # print(embedding.shape)
    # print(name, type(name))

    # break

print(len(audio_embeddings), audio_embeddings[next(iter(audio_embeddings))].shape)
# Save the embeddings to a pickle file
pickle.dump(audio_embeddings, open('videos/whisper.pkl', 'wb'))
