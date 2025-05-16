from __future__ import print_function
import numpy as np
import tensorflow.compat.v1 as tf
from pydub import AudioSegment
import tempfile
import os
import pickle
from tqdm import tqdm
import sys

sys.path.append(os.path.abspath('../models/research/audioset/vggish/'))
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim

# Disable eager execution for compatibility
tf.disable_eager_execution()

def mp4_to_wav(mp4_path):
    """Converts an MP4 file to WAV format using a temporary file."""
    audio = AudioSegment.from_file(mp4_path, format="mp4")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav:
        audio.export(temp_wav.name, format="wav")
        return temp_wav.name  # Return path; file must be manually deleted after use

def extract_vggish_features(mp4_file, sess, features_tensor, embedding_tensor, pproc):
    """Extracts VGGish features from an MP4 file using a preloaded TensorFlow session."""
    wav_file = mp4_to_wav(mp4_file)
    examples_batch = vggish_input.wavfile_to_examples(wav_file)
    os.remove(wav_file)  # Clean up temporary WAV file

    # Run inference
    [embedding_batch] = sess.run([embedding_tensor], feed_dict={features_tensor: examples_batch})
    postprocessed_batch = pproc.postprocess(embedding_batch)
    
    return np.mean(postprocessed_batch, axis=0)

def main():
    files_dir = "_videos/"
    videos = sorted([os.path.join(files_dir, x) for x in os.listdir(files_dir)])
    audio_embeddings = {}

    # Load VGGish model once and keep session open
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, '../models/research/audioset/vggish/vggish_model.ckpt')
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(vggish_params.OUTPUT_TENSOR_NAME)
        pproc = vggish_postprocess.Postprocessor('../models/research/audioset/vggish/vggish_pca_params.npz')

        # Process files sequentially
        for mp4_file in tqdm(videos, desc="Processing video files", total=len(videos)):

            name = os.path.splitext(os.path.basename(mp4_file))[0]
            embedding = extract_vggish_features(mp4_file, sess, features_tensor, embedding_tensor, pproc)
            audio_embeddings[int(name)] = embedding

            break

            # print(embedding.shape)

            # try:
                
            # except Exception as e:
            #     print(e)
            #     print(f'ERROR while processing {mp4_file}')

    # Save embeddings
    print(len(audio_embeddings), audio_embeddings[next(iter(audio_embeddings))].shape)
    with open('videos/vggish.pkl', 'wb') as f:
        pickle.dump(audio_embeddings, f)

if __name__ == "__main__":
    main()
