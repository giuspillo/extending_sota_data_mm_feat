## MMRec

In this folder, we share the code used for our experiments with MMRec. First, we provide the instruction to correctly set up the environment to run the experiments; then, we provide some information to run them.

### Enviroment set up

Hello! Welcome to our tutorial for install a correct virtual environment to run our experiments. We warmly suggest to follow this guide.

First, to run our experiments, we used Python3.11.6 with CUDA 12.3.
Accordingly, we first create a virtual environment and activate it

```
python -m venv _env
source _env/bin/activate
```

You can install the requirements in the requirements.txt file; however, some changes might be needed based on the combination of python version and CUDA driver you are using (in particular, for torch and torch_geometric). 
Due to this, we report here step-by-step the libraries needed to run our experiments, and the commands we run to install them. Note that these libraries are needed for the MMRec framework used in our paper.

- `pandas`: `pip install pandas`
- `torch`: `pip install torch torchvision torchaudio`
- `lmdb`: `pip install lmdb`
- `scipy`: `pip install scipy`
- `yaml`: `pip install PYyaml`
- `matplotlib`: `pip install matplotlib`
- `tqdm: `pip install tqdm`
- `sklearn`: `pip install scikit-learn`
- `torch_geometric`: following [the original documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), you need first to run 
`pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html`
then, you can run
`pip install torch_geometric`
(pay attention at selecting the right version of torch and the right version of cuda driver! They must match those you have installed in your machine. In our case, we install `torch-2.6.0+cu124`, that should be compatible with `cu123`)

That's it! This is everything you need to run our experiments. In our experience, the only problematic package is `torch_geometric`, as it requires compatibility with torch and cuda drivers. After few attempts and documentation study, we were able to run it - we're sure anyone can do the same!

Watch out - In the original [SLMRec implementation](https://github.com/enoche/MMRec/blob/master/src/models/slmrec.py), `torch_scatter` is required, but it is used only in a commented code block. Since `torch_scatter` is another problematic library, and it is never actually used, we commented the import in our code. 

## Running our experiments

To run our script, just go to `src` folder and run:

`python main.py --model=<model_name> --dataset=<dataset_name>`

For example, to run `LATTICE` on `ML1M`:
```
cd src
python main.py --model=movielens_1m --dataset=LATTICE
```

`data` folder contains the `.npy` and the `.inter` files needed to run our experiments, produced during the previous step (Data Processing).

You can edit the configuration by changing parameters in the following files:
- `src/configs/dataset/movielens_1m.yaml`: to change the settings of the model.
- `src/configs/model/LATTICE.yaml`: to change model-specific parameters; in this repo, we set the hyper-parameter values used in our experiments. In this file, for each dataset, you can find the modality names we used to run the experiments, both uni-modal and multi-modal. 
For example, you will find the list of multimodal features (treated as hyper-parameters) for the uni-modal experiments, and the list of multimodal features used for the multi-modal experiments (selecting the best performing ones in the uni-modal setting, as described in the paper).

The same concepts can be generalized to the other models and the other datasets. We did not change the parameters of the models not considered in our experimental session.

At the end of the process, a `log` folder will be created with the results of the experiments.