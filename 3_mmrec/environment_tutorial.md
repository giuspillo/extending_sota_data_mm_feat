Hello! Welcome to our tutorial for install a correct virtual environment to run our experiments. We warmly suggest to follow this guide.

First, to run our experiments, we used Python3.11.6 with CUDA 12.3.
Accordingly, we first create a virtual environment and activate it

python -m venv _env
source _env/bin/activate

You can install the requirements in the requirements.txt file; however, some changes might be needed based on the combination of python version and CUDA driver you are using (in particular, for torch and torch_geometric). 
Due to this, we report here step-by-step the libraries needed to run our experiments, and the commands we run to install them. Note that these libraries are needed for the MMRec framework used in our paper.

- pandas: pip install pandas
- torch: pip install torch torchvision torchaudio
- lmdb: pip install lmdb
- scipy: pip install scipy
- yaml: pip install PYyaml
- matplotlib: pip install matplotlib
- tqdm: pip install tqdm
- sklearn: pip install scikit-learn
- torch_geometric: following [the original documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html), you need first to run 
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
(pay attention at selecting the right version of torch and the right version of cuda driver! They must match those you have installed in your machine. In our case, we install torch-2.6.0+cu124, that should be compatible with cu123)
then, you can run
pip install torch_geometric


That's it! This is everything you need to run our experiments. In our experience, the only problematic package is torch_geometric, as it requires compatibility with torch and cuda drivers. After few attempts and documentation study, we were able to run it - we're sure anyone can do the same!
