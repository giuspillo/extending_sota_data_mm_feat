## Process data

Once multimodal raw files have been download and multimodal features have been encoded, it is time to process such data in a way that is supported by MMRec.

The process is very similar between the datasets, and it is very well documented in the `.ipynb` python notebook that can be found in the folders associated to each dataset.

In a nutshell, here we:
- load interaction and compute core-5
- load extended mappings and multimodal features (`.pkl` format)
- filter out the items that do not have all the features - MMRec requirement
- split interaction data into `train`, `valid`, `test`
- build the `np.array` multimodal feature embeddings, and save them as `.npy` file

Again, we suggest to refer to the notebooks in each dataset folder for more details. Basic python requirements are requested, such as `pandas` or `numpy`.