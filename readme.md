## Repository for the paper entitled "See the Movie, Hear the Song, Read the Book: Extending MovieLens-1M, Last.fm 2K, and DBbook with Multimodal Data"

This repositori is structured in sub-folders, each with their own `readme.md`, which we suggest to carefully read.

## Structure of the repository

As described in the paper, this repository is structred so that anyone can perform:
- download raw multimodal data, for each dataset, and encode them with state-of-the-art encoders
- process the resulting data in a format supported by MMRec
- run the experiment with MMRec to reproduce our results

Each of the actions is mapped with a folder in this repository:
- `1_learn_mm_feat` to download raw files and learn multimodal features
- `2_data_processing` to provide MMRec the files in the supported format
- `3_mmrec` to run the experiments with MMRec

## Learn multimodal features
In `1_learn_mm_feat`, you will find the scripts to download the raw files (for each dataset) and learn the multimodal features using the encoders we selected.

We suggest to carefully read both the `readme.md` file and the `instructions_env.md` file, as they provide crucial information to correctly set up everything. 

In particular, in `instructions_env.md` we describe how to set up the environments needed to obtain the encoded multimodal features. 

All the resource (except the raw data, due to copyright) can also be downloaded from the [Zenodo](https://zenodo.org/records/15403972) repository associated to the paper, on which the resource is released - we share such data there as some of those files exceed the 100MB GitHub size limit.

## Data processing
In `2_data_processing`, you will find the notebook files `.ipynb` to process the encoded multimodal features in a format supported by MMRec. The notebooks are commented and the operations perfomed are simple and intuitive, so to foster the understandability and replicability of our settings. 

As results of this step, you will obtain both `.pkl` and `.json` files, described in the paper and - in addition - downloadable from [Zenodo](https://zenodo.org/records/15403972). Moreover, you will obtain the `.npy` files necessary to run the experiments on MMRec.

## MMRec

In `3_mmrec`, you will find the instructions to set up the environment we used to run our MMRec experiments, together with the data processed in the previous step (in `.npy` format - we were able to upload these files, as they do not exceed the GitHub 100MB limit) and the configuration files we used in our experiments (for both the uni-modal and multi-modal scenarios).

We warmly suggest to carefully read the `readme.md` in this folder as well.