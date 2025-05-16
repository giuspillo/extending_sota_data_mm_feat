# Structure of this folder

In this folder, we provide:
1) the scripts we run to download the raw multimodal data we used (e.g., movie trailers, book covers, music songs)
2) the scripts and the instructions we used to learn the pre-trained multimodal feature, which are one of the most relevant contributions of our paper.

### Download Multimodal Data

First, in each subfolder (representing each dataset) we provide a `download_multimodal_data.ipynb` notebook; these notebooks read the extended mappings we provided, that include the multimodal data raw file links, and download them in the correct format and folder, so to be used in the extraction phase. Some links may be broken due to expired content, as has happened to us for some items (which is why we were unable to provide a raw file for each item for each available mode). 

Such notebooks contain more specific information, to support the reader in downloading the original raw data files.

Once raw files are downloaded, we can extract the multimodal featrues.

### Multimodal feature extraction

First, we report, in the `instructions_env.md`, the instructions to create the virtual environments we used to learn such features. 

In particular, we used two two different environments:
- `_env`, that has been used to learn the multimodal features for all the feature extractors, with the exception of VGGish;
- `_env_vggish`, that has been used only for VGGish. 

The reason is that, at the moment of the submission, the most efficient way to set up an environment compatible with VGGish consists in downloading the model from the [offical repository](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), and following the provided guidelines.

In any case, in our instruction file, we have reported each action we performed to set up such environment. 

In addition, we also report the list of installed packages `pip install <package>`, and, of course, the output of the pip freeze command for both the evironment, `_env` and `_env_vggish`, in `_env_req.txt` and `_env_vggish_req.txt`, respectively.

Now, we discuss the content of each folder.

## MovieLens-1M folder

### Raw source files

The folder `_text` contains the raw text we extracted textual features from, encoded in the `text_ml1m.tsv` file, and downloaded by the `download_multimodal_data.ipynb` notebook.

The folder `_videos` contain the raw videos from which we extracted audio and video features; due to copyright issue, we cannot release the raw files, but we provide in the `data_processing` folder of the this repository the links to the movie trailers we used. Such data has been downloaded by the `download_multimodal_data.ipynb` notebook.

The folder `_posters` contains the raw posters from which we extracted visual features; similarly to the previous case, due to copyright issue, we cannot release the raw files, but we provide in the `data_processing` folder of the this repository the links to the poster image we used.  Such data has been downloaded by the `download_multimodal_data.ipynb` notebook.

### Extraction feature scripts 
This folder contains scripts to run each feature extractor we used; in particular, we have:
- `embedder_text.py`: to extract text features using `all-MiniLM-L6-v2` and `all-mpnet-base-v2`
- `embedder_vit.py`: to extract visual feature using `ViT`
- `embedder_vgg.py`: to extract visual feature using `VGG19`
- `embedder_resnet152.py`: to extract visual feature using `ResNet152`
- `embedder_vggish.py`: to extract audio feature using `VGGish`. 
PAY ATENTION! Only for this extractor you need to use the `_env_vggish` environment!
- `embedder_whisper.py`: to extract audio feature using `Whisper`
- `embedder_r2p1d.py`: to extract video feature using `R(2+1)D`
- `embedder_i3d.py`: to extract visual feature using `I3D`


### Extracted feature files

The folders `text`, `image`, `audio`, `video` contain `.pkl` files that map, for each feature extractor, and for each item ID, the `np.array` associated to that specific modality and extractor.

Note that not every item might be covered by all modalities, so the total dimension of the dictionary can change through different modalities.


## DBbook folder

### Raw source files

The folder `_text` contains the raw text we extracted textual features from, encoded in the `dbbook_text.tsv` file, and downloaded by the `download_multimodal_data.ipynb` notebook.

The folder `_images_` contains the raw book covers from which we extracted visual features; due to copyright issue, we cannot release the raw files, but we provide in the `data_processing` folder of the this repository the links to the poster image we used. Such data has been downloaded by the `download_multimodal_data.ipynb` notebook.

### Extraction feature scripts 
This folder contains scripts to run each feature extractor we used; in particular, we have:
- `embedder_text.py`: to extract text features using `all-MiniLM-L6-v2` and `all-mpnet-base-v2`
- `embedder_vit.py`: to extract visual feature using `ViT`
- `embedder_vgg.py`: to extract visual feature using `VGG19`
- `embedder_resnet152.py`: to extract visual feature using `ResNet152`


### Extracted feature files

The folders `text` and `image` contain `.pkl` files that map, for each feature extractor, and for each item ID, the `np.array` associated to that specific modality and extractor.

Note that not every item might be covered by all modalities, so the total dimension of the dictionary can change through different modalities.



## Last.FM-2K folder

### Raw source files

The folder `_text` contains the raw text we extracted textual features from, encoded in the `lfm2k_text.tsv` file; such data has been obtained by concatenating the user tags associated to each artist (see the `user-taggedartist.dat` file in the original folder of the Last.FM-2K dataset).

The folder `_songs_` contain the raw songs from which we extracted audio and video features; due to copyright issue, we cannot release the raw files, but we provide in the `data_processing` folder of the this repository the links to the movie trailers we used. Such data has been downloaded by the `download_multimodal_data.ipynb` notebook.

The folder `_covers` contains the raw album covers from which we extracted visual features; similarly to the previous case, due to copyright issue, we cannot release the raw files, but we provide in the `data_processing` folder of the this repository the links to the poster image we used. Such data has been downloaded by the `download_multimodal_data.ipynb` notebook.

### Extraction feature scripts 
This folder contains scripts to run each feature extractor we used; in particular, we have:
- `embedder_text.py`: to extract text features using `all-MiniLM-L6-v2` and `all-mpnet-base-v2`
- `embedder_vit.py`: to extract visual feature using `ViT`
- `embedder_vgg.py`: to extract visual feature using `VGG19`
- `embedder_resnet152.py`: to extract visual feature using `ResNet152`
- `embedder_vggish.py`: to extract audio feature using `VGGish`. 
PAY ATENTION! Only for this extractor you need to use the `_env_vggish` environment!
- `embedder_whisper.py`: to extract audio feature using `Whisper`


### Extracted feature files

The folders `text`, `image`, `audio` contain `.pkl` files that map, for each feature extractor, and for each pair artist ID - song, the `np.array` associated to that specific modality and extractor. Note that, while we have a single text associated to each artist, we have more than one songs and album covers associated to the same artist.

For example, for artist with ID `3`, whose `4` popular book covers have been obtained and `2` popular songs have been obtained, we have something similar:

- `<text_extractor.pkl>`: 
    - `3` -> `np.array`

- `<image_extractor.pkl>`: 
    - `3_1` -> `np.array`
    - `3_2` -> `np.array`
    - `3_3` -> `np.array`
    - `3_4` -> `np.array`

- `<audio_extractor.pkl>`: 
    - `3_1` -> `np.array`
    - `3_2` -> `np.array`


Note that not every item might be covered by all modalities, so the total dimension of the dictionary can change through different modalities.


