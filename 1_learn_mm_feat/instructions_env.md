# Virtual Environments information

To perform the feature extraction, we used two `python3.10` environments, whose requirements are reported in `_env_req.txt` and `_env_vggish_req.txt`.

The first one, `_env`, has been built up to extract the multimodal feature for all the extractors, with the exception of `VGGish`, for which another environment, `_env_vggish`; this has been necessary since, at the moment of the submission, the most efficient way to set up an environment compatible with VGGish consists in downloading the model from the [offical repository](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), and following the provided guidelines.

However, to further improve the reproducibility of our results, we provide the list of packages we have installed, in order:

## Environment for all the models except VGGish

```
# create the environment
virtualenv -p python3.10 _env
source _env/bin/activate

# for ViT, ResNet and VGG
pip install torch torchvision torchaudio
pip install transformers
pip install datasets

# for whisper
pip install ffmpeg-python
pip install openai-whisper
pip install pudub

# for text encoders
pip install sentence-transformers
pip install tf-keras

# for R(2+1)D
pip install av

# for I3D
pip install pytorchvideo

```

To run `Whisper` and `R(2+1)D` scripts, use the followings:
```
CUBLAS_WORKSPACE_CONFIG=:4096:8 python embedder_whisper.py
CUBLAS_WORKSPACE_CONFIG=:4096:8 python embedder_r2p1d.py
```

## Environment for VGGish

```
# create the environment
virtualenv -p python3.10 _env_vggish
source _env_vggish/bin/activate
```

The next steps are taken from the official [documentation we followed](https://github.com/tensorflow/models/tree/master/research/audioset/vggish):

In a nutshell, after creating the environment, we need to install the following packages:

```
pip install resampy
pip install tensorflow
pip install tf_slim
pip install six
pip install soundfile

# In addition, we will need
pip install pydub
pip install tqdm

```

Then, download the VGGish repository: 
```
git clone https://github.com/tensorflow/models.git
cd models/research/audioset/vggish
```

Download data files into same directory as code:
```
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
```

Test by running:
```
python vggish_smoke_test.py
```
If it prints "Looks Good To Me!" then VGGish will be succesfully installed!

Finally, you just need to adapt the paths you find in the `embedder_vggish.py` script, so that it points to the directory in which VGGish has been installed (same holds for the `.ckpt` and `.npz` files)